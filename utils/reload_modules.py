from importlib import reload
import sys

def reload_modules(
    module_list: list[str],
) -> None:
    """
    Reload all given modules. Useful for interactive environments like IPython.

    Example usage
    -------------
    .. code-block:: python
        reload_modules(
            [
                rl_models.models.fomaml,
                rl_models.models,
                rl_models,
            ]
        )

    Parameters
    ----------
    module_list: list[str]
        The list containing the names of modules to reload.
    """

    for module in module_list:
        if not module in sys.modules:
            raise ImportError(f"The module {module} is not imported. Import the module first before reloading.")
        reload(sys.modules[module])