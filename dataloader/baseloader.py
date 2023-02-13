class BaseLoader:
    def __int__(self):
        pass

    def load_meta_list(self, root_dir):
        pass

    def load_local(self, root_dir):
        pass

    def load_remote(self, dest_dir):
        pass

    def format(self, data):
        pass

    """
    load single file
    """
    def load(self, file_path):
        pass

    """
    load data from intermediate file
    """
    def load_intermediate(self, file_path, *args):
        pass

