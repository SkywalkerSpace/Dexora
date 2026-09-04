import torch
from transformers import AutoTokenizer, T5EncoderModel


class T5Embedder:
    available_models = [
        "/home/ubuntu/myh/experiment/Dexora/google/t5-v1_1-small",
        "/home/ubuntu/myh/experiment/Dexora/google/t5-v1_1-base",
        "/home/ubuntu/myh/experiment/Dexora/google/t5-v1_1-large",
        "/home/mayuhang/models/t5-v1_1-large",
        "/home/mayuhang/models/t5-v1_1-xxl",
        "google/t5-v1_1-small",
        "google/t5-v1_1-base",
        "google/t5-v1_1-large",
        "google/t5-v1_1-xl",
        "google/t5-v1_1-xxl",
    ]

    def __init__(
        self,
        device,
        from_pretrained=None,
        *,
        cache_dir=None,
        hf_token=None,
        use_text_preprocessing=True,
        t5_model_kwargs=None,
        torch_dtype=None,
        use_offload_folder=None,
        model_max_length=120,
        local_files_only=False,
        quantization_mode=None,
    ):
        self.device = torch.device(device)
        self.torch_dtype = torch_dtype or torch.bfloat16
        self.cache_dir = cache_dir

        if quantization_mode not in (None, "int8", "int4"):
            raise ValueError("quantization_mode must be None, 'int8', or 'int4'")
        if quantization_mode is not None and use_offload_folder is not None:
            raise ValueError("quantization_mode cannot be combined with use_offload_folder")
        if from_pretrained is None:
            raise ValueError("from_pretrained must be a Hugging Face model id or local path")
        # if 'xxl' in from_pretrained:
        #     quantization_mode = 'int4'

        if t5_model_kwargs is None:
            t5_model_kwargs = {
                "low_cpu_mem_usage": True,
                "torch_dtype": self.torch_dtype,
            }

            if quantization_mode is not None:
                try:
                    from transformers import BitsAndBytesConfig
                except ImportError as exc:
                    raise ImportError(
                        "Quantized T5 loading requires bitsandbytes. "
                        "Install it with `pip install bitsandbytes`."
                    ) from exc

                if quantization_mode == "int8":
                    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                else:
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=self.torch_dtype,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                    )
                t5_model_kwargs["quantization_config"] = quantization_config
                t5_model_kwargs["device_map"] = {"": self.device}
            elif use_offload_folder is not None:
                t5_model_kwargs["offload_folder"] = use_offload_folder
                t5_model_kwargs["device_map"] = {
                    "shared": self.device,
                    "encoder.embed_tokens": self.device,
                    "encoder.block.0": self.device,
                    "encoder.block.1": self.device,
                    "encoder.block.2": self.device,
                    "encoder.block.3": self.device,
                    "encoder.block.4": self.device,
                    "encoder.block.5": self.device,
                    "encoder.block.6": self.device,
                    "encoder.block.7": self.device,
                    "encoder.block.8": self.device,
                    "encoder.block.9": self.device,
                    "encoder.block.10": self.device,
                    "encoder.block.11": self.device,
                    "encoder.block.12": "disk",
                    "encoder.block.13": "disk",
                    "encoder.block.14": "disk",
                    "encoder.block.15": "disk",
                    "encoder.block.16": "disk",
                    "encoder.block.17": "disk",
                    "encoder.block.18": "disk",
                    "encoder.block.19": "disk",
                    "encoder.block.20": "disk",
                    "encoder.block.21": "disk",
                    "encoder.block.22": "disk",
                    "encoder.block.23": "disk",
                    "encoder.final_layer_norm": "disk",
                    "encoder.dropout": "disk",
                }
            else:
                t5_model_kwargs["device_map"] = {
                    "shared": self.device,
                    "encoder": self.device,
                }

        self.use_text_preprocessing = use_text_preprocessing
        self.hf_token = hf_token

        self.tokenizer = AutoTokenizer.from_pretrained(
            from_pretrained,
            model_max_length=model_max_length,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        self.model = T5EncoderModel.from_pretrained(
            from_pretrained,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            **t5_model_kwargs,
        ).eval()
        self.is_quantized = bool(
            getattr(self.model, "is_loaded_in_4bit", False)
            or getattr(self.model, "is_loaded_in_8bit", False)
            or getattr(self.model.config, "quantization_config", None) is not None
        )
        self.model_max_length = model_max_length

    def get_text_embeddings(self, texts):
        text_tokens_and_mask = self.tokenizer(
            texts,
            max_length=self.model_max_length,
            padding="longest",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        input_ids = text_tokens_and_mask["input_ids"].to(self.device)
        attention_mask = text_tokens_and_mask["attention_mask"].to(self.device)
        with torch.no_grad():
            text_encoder_embs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )["last_hidden_state"].detach()
        return text_encoder_embs, attention_mask