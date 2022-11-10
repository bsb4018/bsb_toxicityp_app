import sys

class ToxicityException(Exception):
    def __init__(self, error_message, error_detail):
        """
        new ToxicityException
        :param error_message: error message in string format
        """
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message,\
            error_detail=error_detail)

    def __strr__(self):
        return self.error_message

def error_message_detail(error_message, error_detail: sys):
    """
    (exc_type, exc_obj, exc_tb) = (type, value, traceback).
    type : the type of the exception being handled (a subclass of BaseException); 
    value : exception instance (an instance of the exception type); 
    traceback : a traceback object which encapsulates the call stack 
    at the point where the exception originally occurred.
    """
    exc_type, exc_obj, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_file_name #get the file name
    error_message_detailed =  "Error occurred python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name, exc_tb.tb_lineno, str(error_message)
    )

    return error_message_detailed