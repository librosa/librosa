#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''Exception classes for librosa'''


class LibrosaError(Exception):
    '''The root librosa exception class'''
    pass


class LibrosaParameterError(LibrosaError):
    '''Exception class for mal-formed inputs'''
    pass
