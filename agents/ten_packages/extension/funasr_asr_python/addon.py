from ten import (
    Addon,
    register_addon_as_extension,
    TenEnv,
)

@register_addon_as_extension("funasr_asr_python")
class FunASRExtensionAddon(Addon):
    def on_create_instance(self, ten: TenEnv, addon_name: str, context) -> None:
        from .extension import FunASRExtension
        ten.log_info("on_create_instance")
        ten.on_create_instance_done(FunASRExtension(addon_name), context)
