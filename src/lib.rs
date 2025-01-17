#![warn(clippy::pedantic)]
#![warn(clippy::undocumented_unsafe_blocks)]

use std::cell::UnsafeCell;
use std::fmt::{self, Debug, Formatter};
use std::mem::{self, ManuallyDrop};
use std::panic::{resume_unwind, AssertUnwindSafe, RefUnwindSafe, UnwindSafe};
use std::sync::Once;

pub use either::Either;
use void::{ResultVoidExt, Void};

/// A cell which can nominally be modified only once.
///
/// The main difference between this struct and a [`OnceCell`] is that rather than operating on an [`Option`],
/// this type operates on an [`Either`] -- meaning methods such as [`TwiceCell::get_or_init`] take a parameter
/// that can be used to compute the `B` value of a `TwiceCell` based on its `A` value.
///
/// For a thread-safe version of this struct, see [`TwiceLock`].
///
/// [`OnceCell`]: std::cell::OnceCell
///
/// # Examples
///
/// ```
/// use twice_cell::TwiceCell;
///
/// let cell = TwiceCell::new("an initial `A` value for the cell");
/// assert!(cell.get().is_none());
///
/// let value = cell.get_or_init(|s| s.len()); // <- set a `B` value based on the inital `A` value!
/// assert_eq!(*value, 33);
/// assert!(cell.get().is_some());
/// ```
pub struct TwiceCell<A, B> {
    // Invariant: modified at most once, and in the direction from `A` to `B`.
    inner: UnsafeCell<Either<A, B>>,
}

impl<A, B> TwiceCell<A, B> {
    /// Creates a new cell with the given `value`.
    #[inline]
    #[must_use]
    pub const fn new(value: A) -> TwiceCell<A, B> {
        TwiceCell {
            inner: UnsafeCell::new(Either::Left(value)),
        }
    }

    /// Gets the reference to the underlying value.
    ///
    /// Returns `None` if the cell hasn't been set.
    #[inline]
    pub fn get(&self) -> Option<&B> {
        // Safety: we're handing out a reference to B
        match unsafe { self.get_either() } {
            Either::Left(_a) => None,
            // Safety: this pointer is safe to dereference because we got it from `self.get_either`.
            Either::Right(b) => Some(unsafe { &*b }),
        }
    }

    /// Gets the mutable reference to the underlying value.
    ///
    /// Returns `None` if the cell hasn't been set.
    #[inline]
    pub fn get_mut(&mut self) -> Option<&mut B> {
        self.get_either_mut().right()
    }

    /// Gets the reference to either underlying value
    ///
    /// Safety: We can't hand out references to `A` to the user because you could get a reference
    /// to `A` and then mutate `self` through an immutable reference via e.g. `set`.
    ///
    /// However! Since `TwiceCell` is thread-local we _can_ use references to `A` for things like equality checking,
    /// since we know that the single thread can't change the value and check equality at the same time.
    ///
    /// I think.
    #[inline]
    unsafe fn get_either(&self) -> Either<*const A, *const B> {
        // Safety: the caller promises to only ever hand
        // out references to B
        match unsafe { &*self.inner.get() } {
            Either::Left(a) => Either::Left(a),
            Either::Right(b) => Either::Right(b),
        }
    }

    /// Gets the mutable reference to either underlying value.
    #[inline]
    pub fn get_either_mut(&mut self) -> Either<&mut A, &mut B> {
        self.inner.get_mut().as_mut()
    }

    /// Sets the contents of the cell to `value`.
    ///
    /// # Errors
    ///
    /// This method returns `Ok(())` if the cell was empty and `Err(value)` if
    /// it was full.
    ///
    /// # Examples
    ///
    /// ```
    /// use twice_cell::TwiceCell;
    ///
    /// let cell = TwiceCell::new(0);
    /// assert!(cell.get().is_none());
    ///
    /// assert_eq!(cell.set(92), Ok(()));
    /// assert_eq!(cell.set(62), Err(62));
    ///
    /// assert!(cell.get().is_some());
    /// ```
    #[inline]
    pub fn set(&self, value: B) -> Result<(), B> {
        match self.try_insert(value) {
            Ok(_) => Ok(()),
            Err((_, value)) => Err(value),
        }
    }

    /// Sets the contents of the cell to `value` if the cell was empty, then
    /// returns a reference to it.
    ///
    /// # Errors
    ///
    /// This method returns `Ok(&value)` if the cell was empty and
    /// `Err(&current_value, value)` if it was full.
    ///
    /// # Examples
    ///
    /// ```
    /// use twice_cell::TwiceCell;
    ///
    /// let cell = TwiceCell::new(0);
    /// assert!(cell.get().is_none());
    ///
    /// assert_eq!(cell.try_insert(92), Ok(&92));
    /// assert_eq!(cell.try_insert(62), Err((&92, 62)));
    ///
    /// assert!(cell.get().is_some());
    /// ```
    #[inline]
    pub fn try_insert(&self, value: B) -> Result<&B, (&B, B)> {
        if let Some(old) = self.get() {
            return Err((old, value));
        }

        // SAFETY: This is the only place where we set the slot, no races
        // due to reentrancy/concurrency are possible because we're thread-local,
        // and we've checked that slot is currently `A`,
        // so this write maintains the `inner`'s invariant.
        let slot = unsafe { &mut *self.inner.get() };
        *slot = Either::Right(value);

        // Safety: we just set this value.
        let b = unsafe { slot.as_ref().right().unwrap_unchecked() };
        Ok(b)
    }

    /// Gets the contents of the cell, initializing it with `f`
    /// if the cell was unset.
    ///
    /// # Panics
    ///
    /// If `f` panics, the panic is propagated to the caller, and the cell
    /// remains uninitialized.
    ///
    /// It is an error to reentrantly initialize the cell from `f`. Doing
    /// so results in a panic.
    ///
    /// # Examples
    ///
    /// ```
    /// use twice_cell::TwiceCell;
    ///
    /// let cell = TwiceCell::new("hello");
    /// let value = cell.get_or_init(|s| s.len());
    /// assert_eq!(value, &5);
    /// let value = cell.get_or_init(|_| unreachable!());
    /// assert_eq!(value, &5);
    /// ```
    #[inline]
    pub fn get_or_init<F>(&self, f: F) -> &B
    where
        F: FnOnce(&A) -> B,
    {
        self.get_or_try_init(|a| Ok::<B, Void>(f(a))).void_unwrap()
    }

    /// Gets the mutable reference of the contents of the cell,
    /// initializing it with `f` if the cell was unset.
    ///
    /// # Panics
    ///
    /// If `f` panics, the panic is propagated to the caller, and the cell
    /// remains uninitialized.
    ///
    /// # Examples
    ///
    /// ```
    /// use twice_cell::TwiceCell;
    ///
    /// let mut cell = TwiceCell::new("twice_cell");
    /// let value = cell.get_mut_or_init(|s| s.len());
    /// assert_eq!(*value, 10);
    ///
    /// *value += 2;
    /// assert_eq!(*value, 12);
    ///
    /// let value = cell.get_mut_or_init(|_| unreachable!());
    /// assert_eq!(*value, 12);
    /// ```
    #[inline]
    pub fn get_mut_or_init<F>(&mut self, f: F) -> &mut B
    where
        F: FnOnce(&A) -> B,
    {
        self.get_mut_or_try_init(|a| Ok::<B, Void>(f(a)))
            .void_unwrap()
    }

    /// Gets the contents of the cell, initializing it with `f` if
    /// the cell was unset.
    ///
    /// # Panics
    ///
    /// If `f` panics, the panic is propagated to the caller, and the cell
    /// remains uninitialized.
    ///
    /// It is an error to reentrantly initialize the cell from `f`. Doing
    /// so results in a panic.
    ///
    /// # Errors
    ///
    /// If the cell was unset and `f` failed, an
    /// error is returned.
    ///
    /// # Examples
    ///
    /// ```
    /// use twice_cell::TwiceCell;
    ///
    /// let cell = TwiceCell::new(16);
    /// assert_eq!(cell.get_or_try_init(|_| Err(())), Err(()));
    /// assert!(cell.get().is_none());
    /// let value = cell.get_or_try_init(|n| -> Result<i32, ()> {
    ///     Ok(n * 4)
    /// });
    /// assert_eq!(value, Ok(&64));
    /// assert_eq!(cell.get(), Some(&64))
    /// ```
    #[inline]
    pub fn get_or_try_init<F, E>(&self, f: F) -> Result<&B, E>
    where
        F: FnOnce(&A) -> Result<B, E>,
    {
        // Safety: The reference to `A` is only live for the duration of this function.
        match unsafe { self.get_either() } {
            // Safety: this `a` came from `self`
            Either::Left(a) => unsafe { self.try_init(f, a) },
            // Safety: we can always dereference pointers from `self.get_either`
            Either::Right(b) => Ok(unsafe { &*b }),
        }
    }

    /// Gets the mutable reference of the contents of the cell, initializing
    /// it with `f` if the cell was unset.
    ///
    /// # Panics
    ///
    /// If `f` panics, the panic is propagated to the caller, and the cell
    /// remains uninitialized.
    ///
    /// # Errors
    ///
    /// If the cell was unset and `f` failed, an error is returned.
    ///
    /// # Examples
    ///
    /// ```
    /// use twice_cell::TwiceCell;
    ///
    /// let mut cell = TwiceCell::new("not a number!");
    ///
    /// // Failed initializers do not change the value
    /// assert!(cell.get_mut_or_try_init(|s| s.parse::<i32>()).is_err());
    /// assert!(cell.get().is_none());
    ///
    /// let value = cell.get_mut_or_try_init(|_| "1234".parse());
    /// assert_eq!(value, Ok(&mut 1234));
    /// *value.unwrap() += 2;
    /// assert_eq!(cell.get(), Some(&1236))
    /// ```
    #[inline]
    pub fn get_mut_or_try_init<F, E>(&mut self, f: F) -> Result<&mut B, E>
    where
        F: FnOnce(&A) -> Result<B, E>,
    {
        // Safety: The reference to `A` is only live for this function call
        if let Some(a) = unsafe { self.get_either() }.left() {
            // Safety: this `a` came from `self`.
            unsafe { self.try_init(f, a) }?;
        }

        // Safety: we just set this value
        let b = unsafe { self.get_mut().unwrap_unchecked() };
        Ok(b)
    }

    // Avoid inlining the initialization closure into the common path that fetches
    // the already initialized value
    /// Safety: `a` must be non-null and point to an `A` contained in `self`.
    #[cold]
    unsafe fn try_init<F, E>(&self, f: F, a: *const A) -> Result<&B, E>
    where
        F: FnOnce(&A) -> Result<B, E>,
    {
        // Safety: the caller guarantees we can dereference this pointer.
        let val = f(unsafe { &*a })?;
        if let Ok(val) = self.try_insert(val) {
            Ok(val)
        } else {
            panic!("reentrant init")
        }
    }

    /// Consumes the cell, returning the wrapped value.
    ///
    /// Returns `Either::Left(A)` if the cell was unset.
    ///
    /// # Examples
    ///
    /// ```
    /// use twice_cell::TwiceCell;
    /// use either::Either;
    ///
    /// let cell: TwiceCell<u8, &'static str> = TwiceCell::new(123);
    /// assert_eq!(cell.into_inner(), Either::Left(123));
    ///
    /// let cell: TwiceCell<u8, &'static str> = TwiceCell::new(123);
    /// cell.set("hello").unwrap();
    /// assert_eq!(cell.into_inner(), Either::Right("hello"));
    /// ```
    #[inline]
    pub fn into_inner(self) -> Either<A, B> {
        self.inner.into_inner()
    }

    /// Replaces the value in this `TwiceCell`, moving it back to an unset state.
    ///
    /// Safety is guaranteed by requiring a mutable reference.
    ///
    /// # Examples
    ///
    /// ```
    /// use twice_cell::TwiceCell;
    /// use either::Either;
    ///
    /// let mut cell: TwiceCell<i32, &'static str> = TwiceCell::new(123);
    /// assert_eq!(cell.replace(456), Either::Left(123));
    ///
    /// let mut cell = TwiceCell::new(123);
    /// cell.set("goodbye").unwrap();
    /// assert_eq!(cell.replace(456), Either::Right("goodbye"));
    /// assert_eq!(cell.get(), None);
    /// ```
    #[inline]
    pub fn replace(&mut self, a: A) -> Either<A, B> {
        mem::replace(self, TwiceCell::new(a)).into_inner()
    }
}

impl<A: Default, B> Default for TwiceCell<A, B> {
    #[inline]
    fn default() -> Self {
        TwiceCell::new(A::default())
    }
}

impl<A: Debug, B: Debug> Debug for TwiceCell<A, B> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_tuple("TwiceCell")
            // Safety: A borrow of `A` only lives until this function ends.
            .field(&unsafe { self.get_either() })
            .finish()
    }
}

impl<A: Clone, B: Clone> Clone for TwiceCell<A, B> {
    #[inline]
    fn clone(&self) -> Self {
        // Safety: The borrow only lives until mapped to `Clone::clone`
        let inner = match unsafe { self.get_either() } {
            // Safety: This pointer is non-null.
            Either::Left(a) => Either::Left(A::clone(unsafe { &*a })),
            // Safety: This pointer is non-null.
            Either::Right(b) => Either::Right(B::clone(unsafe { &*b })),
        };

        TwiceCell {
            inner: UnsafeCell::new(inner),
        }
    }

    #[inline]
    fn clone_from(&mut self, source: &Self) {
        // Safety: the reference only lives for the rest of this function
        // Note: we can't use `get_either` because we need `&Either<A, B>`, not `Either<&A, &B>`.
        unsafe { &*source.inner.get() }.clone_into(self.inner.get_mut());
    }
}

impl<A: PartialEq, B: PartialEq> PartialEq for TwiceCell<A, B> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        // Safety: the references only live for the duration of this function.
        unsafe { self.get_either() == other.get_either() }
    }
}

impl<A: Eq, B: Eq> Eq for TwiceCell<A, B> {}

impl<A, B> From<B> for TwiceCell<A, B> {
    /// Creates a new `TwiceCell<A, B>` which is already set to the given `value`.
    #[inline]
    fn from(value: B) -> Self {
        TwiceCell {
            inner: UnsafeCell::new(Either::Right(value)),
        }
    }
}

#[cfg(feature = "serde")]
impl<A: serde::Serialize, B: serde::Serialize> serde::Serialize for TwiceCell<A, B> {
    #[inline]
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        // Safety: The reference only lives for the duration of this function.
        either::for_both!(unsafe { self.get_either() }, ptr => unsafe { &*ptr }.serialize(serializer))
    }
}

#[cfg(feature = "serde")]
impl<'de, A: serde::Deserialize<'de>, B: serde::Deserialize<'de>> serde::Deserialize<'de>
    for TwiceCell<A, B>
{
    #[inline]
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let inner = Either::deserialize(deserializer)?;
        Ok(TwiceCell {
            inner: UnsafeCell::new(inner),
        })
    }
}

/// [`Either`], but without the tag on the union.
///
/// Requires some outside source of synchronization to figure out which field is currently set.
/// In [`TwiceLock`], that's [`Once`].
union UntaggedEither<A, B> {
    a: ManuallyDrop<A>,
    b: ManuallyDrop<B>,
}

impl<A, B> UntaggedEither<A, B> {
    /// Construct a new `UntaggedEither` with the given value.
    #[inline]
    const fn new(value: A) -> UntaggedEither<A, B> {
        UntaggedEither {
            a: ManuallyDrop::new(value),
        }
    }

    /// Take `A` out of this `UntaggedEither`, consuming it.
    ///
    /// Safety: The caller must guarantee that field `a` is currently set.
    #[inline]
    unsafe fn into_a(self) -> A {
        ManuallyDrop::into_inner(self.a)
    }

    /// Take `B` out of this `UntaggedEither`, consuming it.
    ///
    /// Safety: The caller must guarantee that field `b` is currently set.
    #[inline]
    unsafe fn into_b(self) -> B {
        ManuallyDrop::into_inner(self.b)
    }
}

impl<A, B> From<B> for UntaggedEither<A, B> {
    /// Construct a new `UntaggedEither` already set to `B`.
    fn from(value: B) -> Self {
        Self {
            b: ManuallyDrop::new(value),
        }
    }
}

pub struct TwiceLock<A, B> {
    /// Synchronization primitive used to enforce whether field `a` or `b` is set in `.value`.
    once: Once,

    // Whether or not the value is set is tracked by `once.is_completed()`,
    // Invariant: modified at most once, and in the direction from `A` to `B`.
    value: UnsafeCell<UntaggedEither<A, B>>,
}

impl<A, B> TwiceLock<A, B> {
    /// Creates a new empty cell.
    #[inline]
    #[must_use]
    pub const fn new(value: A) -> TwiceLock<A, B> {
        TwiceLock {
            once: Once::new(),
            value: UnsafeCell::new(UntaggedEither::new(value)),
        }
    }

    /// Gets the reference to the underlying value.
    ///
    /// Returns `None` if the cell is unset, or being set.
    /// This method never blocks.
    #[inline]
    pub fn get(&self) -> Option<&B> {
        if self.is_set() {
            // Safety: we checked `is_set`.
            Some(unsafe { &*self.get_b_unchecked() })
        } else {
            None
        }
    }

    /// Gets the mutable reference to the underlying value.
    ///
    /// Returns `None` if the cell is unset. This method never blocks.
    #[inline]
    pub fn get_mut(&mut self) -> Option<&mut B> {
        if self.is_set() {
            // Safety: we checked `is_set`.
            Some(unsafe { self.get_b_unchecked_mut() })
        } else {
            None
        }
    }

    /// Get a reference to either underlying value.
    ///
    /// Safety: an `A` reference can only be used to initialize the value, since as soon as the value
    /// is set the reference will be invalidated.
    ///
    /// Unlike `TwiceCell`, in which `A` references can be used for other purposes (e.g. equality checking),
    /// we can _only_ use `&A` to set `self.value` inside of a `Once` closure. This is because if on
    /// one thread we're using `A` to e.g. check equality, another thread could mutate `self` into `B`,
    /// causing the reference to `A` to be invalidated.
    #[inline]
    unsafe fn get_either(&self) -> Either<*const A, *const B> {
        if self.is_set() {
            // Safety: we're initialized, therefore `b` is set.
            Either::Right(unsafe { self.get_b_unchecked() })
        } else {
            // Safety: we're uninitialized, therefore `a` is set.
            Either::Left(unsafe { self.get_a_unchecked() })
        }
    }

    /// Gets the mutable reference to either underlying value.
    //
    // Unlike `get_either`, this function is perfectly safe because we have mutable access.
    // No other threads can mutate this value while we hold a mutable reference.
    #[inline]
    pub fn get_either_mut(&mut self) -> Either<&mut A, &mut B> {
        if self.is_set() {
            // Safety: we're initialized, therefore `b` is set.
            Either::Right(unsafe { self.get_b_unchecked_mut() })
        } else {
            // Safety: we're uninitialized, therefore `a` is set.
            Either::Left(unsafe { self.get_a_unchecked_mut() })
        }
    }

    /// Sets the contents of this cell to `value`.
    ///
    /// May block if another thread is currently attempting to initialize the cell. The cell is
    /// guaranteed to contain a value when set returns, though not necessarily the one provided.
    ///
    /// # Errors
    ///
    /// Returns `Ok(())` if the cell's value was set by this call.
    ///
    /// # Examples
    ///
    /// ```
    /// use twice_cell::TwiceLock;
    ///
    /// static CELL: TwiceLock<&str, i32> = TwiceLock::new("initial value");
    ///
    /// fn main() {
    ///     assert!(CELL.get().is_none());
    ///
    ///     std::thread::spawn(|| {
    ///         assert_eq!(CELL.set(92), Ok(()));
    ///     }).join().unwrap();
    ///
    ///     assert_eq!(CELL.set(62), Err(62));
    ///     assert_eq!(CELL.get(), Some(&92));
    /// }
    /// ```
    pub fn set(&self, value: B) -> Result<(), B>
    where
        A: RefUnwindSafe,
    {
        match self.try_insert(value) {
            Ok(_) => Ok(()),
            Err((_, value)) => Err(value),
        }
    }

    /// Sets the contents of this cell to `value` if the cell was empty, then
    /// returns a reference to it.
    ///
    /// May block if another thread is currently attempting to initialize the cell. The cell is
    /// guaranteed to contain a value when set returns, though not necessarily the one provided.
    ///
    /// # Errors
    ///
    /// Returns `Ok(&value)` if the cell was empty and `Err(&current_value, value)` if it was full.
    ///
    /// # Examples
    ///
    /// ```
    ///
    /// use twice_cell::TwiceLock;
    ///
    /// static CELL: TwiceLock<&'static str, i32> = TwiceLock::new("initial value");
    ///
    /// fn main() {
    ///     assert!(CELL.get().is_none());
    ///
    ///     std::thread::spawn(|| {
    ///         assert_eq!(CELL.try_insert(92), Ok(&92));
    ///     }).join().unwrap();
    ///
    ///     assert_eq!(CELL.try_insert(62), Err((&92, 62)));
    ///     assert_eq!(CELL.get(), Some(&92));
    /// }
    /// ```
    #[inline]
    pub fn try_insert(&self, value: B) -> Result<&B, (&B, B)>
    where
        A: RefUnwindSafe,
    {
        let mut value = Some(value);
        let mut safe_value = AssertUnwindSafe(&mut value);
        // Safety: the value is set to `Some` right above
        let res = self.get_or_init(move |_| unsafe { safe_value.take().unwrap_unchecked() });

        match value {
            None => Ok(res),
            Some(value) => Err((res, value)),
        }
    }

    /// Gets the contents of the cell, initializing it with `f` if the cell
    /// was empty.
    ///
    /// Many threads may call `get_or_init` concurrently with different
    /// initializing functions, but it is guaranteed that only one function
    /// will be executed.
    ///
    /// # Panics
    ///
    /// If `f` panics, the panic is propagated to the caller, and the cell
    /// remains uninitialized.
    ///
    /// It is an error to reentrantly initialize the cell from `f`. The
    /// exact outcome is unspecified. Current implementation deadlocks, but
    /// this may be changed to a panic in the future.
    ///
    /// # Examples
    ///
    /// ```
    /// use twice_cell::TwiceLock;
    ///
    /// let cell = TwiceLock::new("initial value");
    /// let value = cell.get_or_init(|s| s.len());
    /// assert_eq!(value, &13);
    /// let value = cell.get_or_init(|_| unreachable!());
    /// assert_eq!(value, &13);
    /// ```
    #[inline]
    pub fn get_or_init<F>(&self, f: F) -> &B
    where
        F: UnwindSafe + FnOnce(&A) -> B,
        A: RefUnwindSafe,
    {
        self.get_or_try_init(|a| Ok::<B, Void>(f(a))).void_unwrap()
    }

    /// Gets the mutable reference of the contents of the cell, initializing
    /// it with `f` if the cell was empty.
    ///
    /// Many threads may call `get_mut_or_init` concurrently with different
    /// initializing functions, but it is guaranteed that only one function
    /// will be executed.
    ///
    /// # Panics
    ///
    /// If `f` panics, the panic is propagated to the caller, and the cell
    /// remains uninitialized.
    ///
    /// # Examples
    ///
    /// ```
    /// use twice_cell::TwiceLock;
    ///
    /// let mut cell = TwiceLock::new("initial value");
    /// let value = cell.get_mut_or_init(|s| s.len());
    /// assert_eq!(*value, 13);
    ///
    /// *value += 2;
    /// assert_eq!(*value, 15);
    ///
    /// let value = cell.get_mut_or_init(|_| unreachable!());
    /// assert_eq!(*value, 15);
    /// ```
    #[inline]
    pub fn get_mut_or_init<F>(&mut self, f: F) -> &mut B
    where
        F: UnwindSafe + FnOnce(&A) -> B,
        A: RefUnwindSafe,
    {
        self.get_mut_or_try_init(|a| Ok::<B, Void>(f(a)))
            .void_unwrap()
    }

    /// Gets the contents of the cell, initializing it with `f` if
    /// the cell was empty.
    ///
    /// # Panics
    ///
    /// If `f` panics, the panic is propagated to the caller, and
    /// the cell remains uninitialized.
    ///
    /// # Errors
    ///
    /// If the cell was empty and `f` failed, an error is returned.
    ///
    /// It is an error to reentrantly initialize the cell from `f`.
    /// The exact outcome is unspecified. Current implementation
    /// deadlocks, but this may be changed to a panic in the future.
    ///
    /// # Examples
    ///
    /// ```
    /// use twice_cell::TwiceLock;
    ///
    /// let cell = TwiceLock::new("initial value");
    ///
    /// assert_eq!(cell.get_or_try_init(|_| Err(())), Err(()));
    /// assert!(cell.get().is_none());
    ///
    /// let value = cell.get_or_try_init(|s| -> Result<usize, ()> {
    ///     Ok(s.len())
    /// });
    /// assert_eq!(value, Ok(&13));
    /// assert_eq!(cell.get(), Some(&13))
    /// ```
    #[inline]
    pub fn get_or_try_init<F, E>(&self, f: F) -> Result<&B, E>
    where
        F: UnwindSafe + FnOnce(&A) -> Result<B, E>,
        E: Send + 'static,
        A: RefUnwindSafe,
    {
        // Fast path check
        // NOTE: We need to perform an acquire on the state in this method
        // in order to correctly synchronize `LazyLock::force`. This is
        // currently done by calling `self.get_either()`, which in turn calls
        // `self.is_initialized()`, which in turn performs the acquire.
        //
        // Safety: the `A` reference is used to initialize `self.value`
        match unsafe { self.get_either() } {
            // Safety: `a` came from `self`.
            Either::Left(a) => unsafe { self.initialize(f, a) }?,
            // Safety: This pointer is non-null because it came from `self.get_either()`
            Either::Right(b) => return Ok(unsafe { &*b }),
        };

        debug_assert!(self.is_set());
        // SAFETY: The inner value has been initialized
        Ok(unsafe { &*self.get_b_unchecked() })
    }

    /// Gets the mutable reference of the contents of the cell, initializing
    /// it with `f` if the cell was empty.
    ///
    /// # Panics
    ///
    /// If `f` panics, the panic is propagated to the caller, and
    /// the cell remains uninitialized.
    ///
    /// # Errors
    ///
    /// If the cell was empty and `f` failed, an error is returned.
    ///
    /// # Examples
    ///
    /// ```
    /// use twice_cell::TwiceLock;
    ///
    /// let mut cell: TwiceLock<&'static str, usize> = TwiceLock::new("not a number");
    ///
    /// // Failed initializers do not change the value
    /// assert!(cell.get_mut_or_try_init(|s| s.parse()).is_err());
    /// assert!(cell.get().is_none());
    ///
    /// let value = cell.get_mut_or_try_init(|_| "1234".parse());
    /// assert_eq!(value, Ok(&mut 1234));
    ///
    /// *value.unwrap() += 2;
    /// assert_eq!(cell.get(), Some(&1236));
    /// ```
    #[inline]
    pub fn get_mut_or_try_init<F, E>(&mut self, f: F) -> Result<&mut B, E>
    where
        F: UnwindSafe + FnOnce(&A) -> Result<B, E>,
        E: Send + 'static,
        A: RefUnwindSafe,
    {
        // Safety: we're only using `a` to initialize
        if let Some(a) = unsafe { self.get_either() }.left() {
            // Safety: `a` came from `self`.
            unsafe { self.initialize(f, a) }?;
        }

        debug_assert!(self.is_set());
        // Safety: The inner value has been initialized
        Ok(unsafe { self.get_b_unchecked_mut() })
    }

    /// Takes the value out of this `TwiceLock`, moving it back to an unset state.
    ///
    /// Returns `Either::Left` if the `TwiceLock` hasn't been set.
    ///
    /// Safety is guaranteed by requiring a mutable reference.
    ///
    /// # Examples
    ///
    /// ```
    /// use twice_cell::TwiceLock;
    /// use either::Either;
    ///
    /// let mut cell: TwiceLock<i32, String> = TwiceLock::new(123);
    /// assert_eq!(cell.replace(456), Either::Left(123));
    ///
    /// let mut cell = TwiceLock::new(123);
    /// cell.set("hello").unwrap();
    ///
    /// assert_eq!(cell.replace(456), Either::Right("hello"));
    /// assert_eq!(cell.get(), None);
    /// ```
    #[inline]
    pub fn replace(&mut self, value: A) -> Either<A, B> {
        let inner = mem::replace(self.value.get_mut(), UntaggedEither::new(value));

        if self.is_set() {
            self.once = Once::new();
            // SAFETY: `self.value` is initialized and contains a valid `B`.
            // `self.once` is reset, so `is_initialized()` will be false again
            // which prevents the value from being read twice.
            Either::Right(unsafe { inner.into_b() })
        } else {
            // Safety: we know `self` is uninitialized
            Either::Left(unsafe { inner.into_a() })
        }
    }

    #[inline]
    fn is_set(&self) -> bool {
        self.once.is_completed()
    }

    /// Safety: `a` must be non-null and point to the value contained in `self`.
    #[cold]
    unsafe fn initialize<F, E>(&self, f: F, a: *const A) -> Result<(), E>
    where
        F: FnOnce(&A) -> Result<B, E>,
        E: Send + 'static,
        A: RefUnwindSafe,
        F: UnwindSafe,
    {
        let slot = &self.value;

        // Since we don't have access to `p.poison()`, we have to panic and then catch it explicitly.
        std::panic::catch_unwind(AssertUnwindSafe(|| {
            self.once.call_once_force(|_| {
                // Safety: the caller guarantees this pointer is non-null.
                match f(unsafe { &*a }) {
                    Ok(value) => {
                        // Safety: we have unique access to the slot because we're inside a `once` closure.
                        unsafe { (*slot.get()).b = ManuallyDrop::new(value) };
                    }
                    Err(e) => resume_unwind(Box::new(e)),
                }
            });
        }))
        .map_err(|any| match any.downcast() {
            Ok(e) => *e,
            Err(any) => resume_unwind(any),
        })
    }

    /// # Safety
    ///
    /// The value must be unset.
    #[inline]
    unsafe fn get_a_unchecked(&self) -> *const A {
        debug_assert!(!self.is_set());

        // Safety: The caller upholds the contract that the value is unset, and therefore `.a` is set.
        unsafe { &*(*self.value.get()).a }
    }

    #[inline]
    unsafe fn get_a_unchecked_mut(&mut self) -> &mut ManuallyDrop<A> {
        debug_assert!(!self.is_set());

        // Safety: the caller upholds the contract that the value is unset, and therefore `.a` is set.
        unsafe { &mut self.value.get_mut().a }
    }

    /// # Safety
    ///
    /// The value must be set.
    #[inline]
    unsafe fn get_b_unchecked(&self) -> *const B {
        debug_assert!(self.is_set());

        // Safety: The caller upholds the contract that the value (and therefore `.b`) is set.
        unsafe { &*(*self.value.get()).b }
    }

    /// # Safety
    ///
    /// The value must be set.
    #[inline]
    unsafe fn get_b_unchecked_mut(&mut self) -> &mut ManuallyDrop<B> {
        debug_assert!(self.is_set());

        // Safety: The caller upholds the contract that the value (and therefore `.b`) is set.
        unsafe { &mut self.value.get_mut().b }
    }
}

impl<A, B> Drop for TwiceLock<A, B> {
    #[inline]
    fn drop(&mut self) {
        if self.is_set() {
            // Safety: we know the data was initialized and therefore `b` is set.
            let b = unsafe { self.get_b_unchecked_mut() };

            // Safety: we're dropping in the destructor.
            unsafe { ManuallyDrop::drop(b) };
        } else {
            // Safety: the data is uninitialized and therefore `a` is set.
            let a = unsafe { self.get_a_unchecked_mut() };

            // Safety: we're dropping in the destructor.
            unsafe { ManuallyDrop::drop(a) };
        }
    }
}

/// Safety: The `UnsafeCell` exists to enforce borrow checking via the `Once` primitive instead of the compiler;
/// that is to say, we can override `UnsafeCell`'s `!Send`-ness if both `A` and `B` are `Send`.
unsafe impl<A: Send, B: Send> Send for TwiceLock<A, B> {}

/// Safety: The same is true for `Sync`, except that we also need `A` and `B` to `impl Send` for the same reason
/// there's `Send` bounds on the [`OnceLock`](`std::sync::OnceLock`) implementation:
///
/// > Why do we need `T: Send`?
/// > Thread A creates a `OnceLock` and shares it with scoped thread B,
/// > which fills the cell, which is then destroyed by A.
/// > That is, destructor observes a sent value.
unsafe impl<A: Sync + Send, B: Sync + Send> Sync for TwiceLock<A, B> {}

impl<A: Default, B> Default for TwiceLock<A, B> {
    #[inline]
    fn default() -> Self {
        TwiceLock::new(A::default())
    }
}

impl<A: Debug, B: Debug> Debug for TwiceLock<A, B> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let mut d = f.debug_tuple("TwiceLock");
        match self.get() {
            Some(v) => d.field(v),
            None => d.field(&format_args!("<unset>")),
        };

        d.finish()
    }
}

impl<A, B> From<B> for TwiceLock<A, B> {
    /// Create a new cell with its contents set to `value`.
    ///
    /// # Example
    ///
    /// ```
    /// use std::sync::OnceLock;
    ///
    /// # fn main() -> Result<(), i32> {
    /// let a = OnceLock::from(3);
    /// let b = OnceLock::new();
    /// b.set(3)?;
    /// assert_eq!(a, b);
    /// Ok(())
    /// # }
    /// ```
    #[inline]
    fn from(value: B) -> Self {
        let once = Once::new();
        once.call_once(|| {});

        let value = UnsafeCell::new(UntaggedEither::from(value));

        TwiceLock {
            once,
            value,
        }
    }
}
