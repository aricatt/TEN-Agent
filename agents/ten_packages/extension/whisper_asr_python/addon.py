#
#
# Agora Real Time Engagement
# Created by Wei Hu in 2024-08.
# Copyright (c) 2024 Agora IO. All rights reserved.
#
#
from ten import (
    Addon,
    register_addon_as_extension,
    TenEnv,
)


@register_addon_as_extension("whisper_asr_python")
class WhisperASRExtensionAddon(Addon):

    def on_create_instance(self, ten: TenEnv, addon_name: str, context) -> None:
        from .extension import WhisperASRExtension
        ten.log_info("on_create_instance")
        ten.on_create_instance_done(WhisperASRExtension(addon_name), context)
