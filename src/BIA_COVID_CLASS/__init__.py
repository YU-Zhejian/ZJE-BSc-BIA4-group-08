try:
    import sklearnex

    sklearnex.patch_sklearn()
except ImportError:
    sklearnex = None
