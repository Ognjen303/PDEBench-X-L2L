import pdebench.models

print('Imported pdebench sucsessfully!')

from pkgutil import iter_modules

def list_submodules(module):
    for submodule in iter_modules(module.__path__):
        print(submodule.name)

list_submodules(pdebench)