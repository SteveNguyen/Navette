//! Parameter validation: every parameter struct has a `validate()` returning a [`ParamError`] that names
//! the field, and the constructors of the core types call it. A bad value is an error where the object is
//! built, not a hang (an arc sampled with step = 0), a huge allocation (a grid of size / 0 cells) or a check
//! that silently never fires (a NaN compared with a timer) later.

/// An invalid parameter value: which one, and what is expected.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParamError(pub String);

impl std::fmt::Display for ParamError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for ParamError {}

/// `value` must be finite and > 0.
pub(crate) fn positive(name: &str, value: f64) -> Result<(), ParamError> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(ParamError(format!("{name} = {value}: must be finite and > 0")))
    }
}

/// `value` must be finite and >= 0.
pub(crate) fn non_negative(name: &str, value: f64) -> Result<(), ParamError> {
    if value.is_finite() && value >= 0.0 {
        Ok(())
    } else {
        Err(ParamError(format!("{name} = {value}: must be finite and >= 0")))
    }
}

/// `value` must be finite.
pub(crate) fn finite(name: &str, value: f64) -> Result<(), ParamError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(ParamError(format!("{name} = {value}: must be finite")))
    }
}

/// `value` must be in [1, max].
pub(crate) fn count(name: &str, value: usize, max: usize) -> Result<(), ParamError> {
    if (1..=max).contains(&value) {
        Ok(())
    } else {
        Err(ParamError(format!("{name} = {value}: must be in 1..={max}")))
    }
}
