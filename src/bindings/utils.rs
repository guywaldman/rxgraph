#[macro_export]
macro_rules! py_dataclass {
    (
        $(#[$meta:meta])*
        $vis:vis struct $name:ident {
            $(
                $field:ident : $ty:ty
            ),* $(,)?
        }
    ) => {
        #[::pyo3::pyclass(from_py_object)]
        #[derive(Debug, Clone)]
        $(#[$meta])*
        $vis struct $name {
            $(
                #[pyo3(get, set)]
                pub $field: $ty,
            )*
        }

        #[::pyo3::pymethods]
        impl $name {
            #[new]
            pub fn new($($field: $ty),*) -> Self {
                Self { $($field),* }
            }

            pub fn __repr__(&self) -> String {
                let fields = [
                    $(
                        ::std::format!(
                            "{}={:?}",
                            ::std::stringify!($field),
                            &self.$field
                        ),
                    )*
                ];

                ::std::format!(
                    "{}({})",
                    ::std::stringify!($name),
                    fields.join(", ")
                )
            }
        }
    };
}
