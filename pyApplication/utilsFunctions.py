from Constants import Constants
import numpy as np

# currently not used
def debug(func):

    def printArgsInfo(*args,**kwargs):
        try:
            return func(*args,**kwargs)
        except Exception as e:
            print("="*10 + " Function '{}' failed ".format(func.__name__) + "="*10)
            print("="*5 +  " args given   " + "="*5)
            for arg in args:
                print(type(arg)) if type(arg) != np.ndarray else print((type(arg),arg.shape))
            print("=" * 5 + " kwargs given " + "=" * 5)
            for k in kwargs.keys():
                val = kwargs[k]
                print(k,type(val)) if type(val) != np.ndarray else print(k,type(val),val.shape)
            raise Exception(e)

    return printArgsInfo

def inputOutputValidation(func):

    def prePostfunc(*args,**kwargs):

        if func.__name__ in validationFunctions.inputValidationDict:
            preFunc = validationFunctions.inputValidationDict[func.__name__]
            print("running '{}' input validation on '{}'".format(preFunc.__name__, func.__name__)) if Constants.VERBOSE else None
            # pre functions accept same arguments as original function
            preFunc(*args, **kwargs)

        ret = func(*args, **kwargs)

        if func.__name__ in validationFunctions.outputValidationDict:
            postFunc = validationFunctions.outputValidationDict[func.__name__]
            print("running '{}' input validation on '{}'".format(postFunc.__name__,func.__name__)) if Constants.VERBOSE else None
            # post functions accept return value of original function
            postFunc(ret)

        return ret

    return prePostfunc

"""
validation functions to be used in 'validationFunctions' class
"""
def checkShape(ndArr):
    print( ndArr.shape) if Constants.VERBOSE else None

def minMaxArray(ndArr):
    print( np.max(ndArr) , np.min(ndArr) ) if Constants.VERBOSE else None


"""
use 'inputOutputValidation' decorator to validate a function
in the following dictionaries key is the function to validate (the decorated fucntion), value that performs the validations:
'inputValidationDict' is for input validation funcs.these functions accept same arguments as validated function.
'outputValidationDict' is for output validation. these functions accept return value of validated function.
"""
class validationFunctions():

    inputValidationDict = {}
    # iputValidationDict['grayscale'] = checkShape

    outputValidationDict = {}
    # outputValidationDict['subtract'] = minMaxArray

# Todo - separate to another file?
class targetFunctions():

    @staticmethod
    def printClassification(probs):
        if probs[0][0] > probs[0][1]:
            print("DOG! " + str(probs[0][0]))
        else:
            print("HUMAN! " + str(probs[0][1]))