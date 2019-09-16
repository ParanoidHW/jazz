from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division

import os


JAZZ_API_NAME = 'jazz'


class Expose:
    def __init__(self, *args, **kwargs):
        """

        :param args: API names in dot delimited format
        :param kwargs: Optional keyed arguments
            api_name: Name of the API you want to generate. Default to 'jazz'
        """
        self._names = args
        self._api_name = kwargs.get('api_name', JAZZ_API_NAME)

    def __call__(self, func):
        pass

    def do_expose(self, func_dict):
        """
        Expose the functions in func_dict values using the corresponding keys.
        :param func_dict:
        :return:
        """


