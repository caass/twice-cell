#![warn(clippy::pedantic)]
#![warn(clippy::undocumented_unsafe_blocks)]

use std::cell::UnsafeCell;
use std::fmt;
use std::mem;

pub use either::Either;
use void::{ResultVoidExt, Void};

pub struct TwiceCell<A, B> {
    // Invariant: modified at most once.
    inner: UnsafeCell<Either<A, B>>,
}

impl<A, B> TwiceCell<A, B> {
    /// Creates a new cell.
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
        unsafe { self.get_either() }.right()
    }

    /// Gets the mutable reference to the underlying value.
    ///
    /// Returns `None` if the cell hasn't been set.
    #[inline]
    pub fn get_mut(&mut self) -> Option<&mut B> {
        self.get_either_mut().right()
    }

    /// Safety: Any reference to `A` cannot be handed out, because if the caller
    /// holds on to that reference, then mutates `self`, that reference would be invalid.
    ///
    /// Any references to `A` need to be consumed internally before the function returns.
    /// Normally we'd just mark this as utterly unsafe period but since we're thread-local
    /// we don't need to worry about it.
    #[inline]
    unsafe fn get_either(&self) -> Either<&A, &B> {
        // Safety: the caller promises to only ever hand
        // out references to B
        unsafe { &*self.inner.get() }.as_ref()
    }

    /// Gets the mutable reference to either underyling value.
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
            Either::Left(a) => self.try_init(f, a),
            Either::Right(b) => Ok(b),
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
            self.try_init(f, a)?;
        }

        // Safety: we just set this value
        let b = unsafe { self.get_mut().unwrap_unchecked() };
        Ok(b)
    }

    // Avoid inlining the initialization closure into the common path that fetches
    // the already initialized value
    #[cold]
    fn try_init<F, E>(&self, f: F, a: &A) -> Result<&B, E>
    where
        F: FnOnce(&A) -> Result<B, E>,
    {
        let val = f(a)?;
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

impl<A: fmt::Debug, B: fmt::Debug> fmt::Debug for TwiceCell<A, B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
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
        let inner = unsafe { self.get_either() }.map_either(A::clone, B::clone);

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
        unsafe { self.get_either() }.serialize(serializer)
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
