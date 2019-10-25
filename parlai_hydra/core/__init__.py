def setup_env():
    import os
    from os.path import dirname, realpath, join

    parlai_home = dirname(dirname(dirname(realpath(__file__))))

    def set_default_env(key, val):
        """
        Sets the value for environment variable if not already set
        """
        if key not in os.environ:
            os.environ[key] = val

    set_default_env('PARLAI_HOME', parlai_home)
    set_default_env('PARLAI_DATAPATH', join(parlai_home, 'data'))
    set_default_env('PARLAI_DATAPATH', join(parlai_home, 'download'))


setup_env()
