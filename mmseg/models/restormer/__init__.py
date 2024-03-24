from .prompt_restormer import PromptRestormer



__all__ = ['PromptRestormer']

custom_imports = dict(
    imports=['mmseg.models.restormer.prompt_restormer'],
    allow_failed_imports=False)