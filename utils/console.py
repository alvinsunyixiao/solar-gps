import sys

# see https://stackoverflow.com/questions/287871/how-to-print-colored-text-in-terminal-in-python
class TermColor:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def is_color_supported():
    return hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()

def make_printer(color):
    def printer(msg, **kwargs):
        if is_color_supported():
            msg = color + msg + TermColor.ENDC
        print(msg, **kwargs)
    return printer

print_ok = make_printer(TermColor.OKGREEN)
print_info = make_printer(TermColor.OKBLUE)
print_warn = make_printer(TermColor.WARNING)
print_fail = make_printer(TermColor.FAIL)
