from copy import deepcopy
import torch
from warnings import warn
from projects.style_gen.classifier import ClassifierAgent
from parlai_internal.tasks.empathy_labeled_utterances.agents import FUTURE_SEP_TOKEN

# This agent is for fine-tuning pretrained classifiers that have a (potentially) different classifier head dimension.
# It reformats the model and optimizer to state dict by randomly initializing the classifer weight and bias.


class FromPreTrainedClassifierAgent(ClassifierAgent):
    @staticmethod
    def add_cmdline_args(parser):
        ClassifierAgent.add_cmdline_args(parser)
        parser = parser.add_argument_group("From Pre-trained Classifier Arguments")
        parser.add_argument(
            "--init-new-optim",
            type="bool",
            default=False,
            help="re-initialize optimizer instead of using stored history",
        )
        # for separating previous and future utterance
        parser.set_defaults(special_tok_list=FUTURE_SEP_TOKEN)

    def load_state_dict(self, state_dict):
        modified_keys = set()
        for key, val in self.model.state_dict().items():
            if val.shape != state_dict[key].shape:
                warn(
                    f"Missing key: {key} has different size in model; "
                    f"this behavior is expected if replacing classifier head from state_dict"
                )
                state_dict[key] = val  # replace with initialized weight
                modified_keys.add(key)
        self.model.load_state_dict(state_dict)

        try:
            self.model.load_state_dict(state_dict)
        except RuntimeError as msg:
            msg_ = str(msg)
            if "size mismatch" in msg_ and "embedding" in msg_:
                if hasattr(self, "special_toks") and len(self.special_toks) > 0:
                    state_dict = self._resize_token_embeddings(state_dict, msg_)
                    self.model.load_state_dict(state_dict)
                    self.resized_embeddings = True  # make note that we resized here
                else:
                    raise RuntimeError(
                        f"{msg_}\n"
                        "-----------------\n"
                        "Could not load the model due to a size mismatch in the "
                        "embeddings. A common reason for this is trying to load "
                        "a model trained with fp16 but loaded without fp16. Try "
                        "adding --fp16 true or --force-fp16-tokens true."
                    )
            else:
                raise
        return modified_keys

    def load(self, path):
        """
        Return opt and model states.
        Override this method for more specific loading.
        """
        import parlai.utils.pickle
        from parlai.utils.io import PathManager
        from itertools import chain

        with PathManager.open(path, "rb") as f:
            states = torch.load(
                f, map_location=lambda cpu, _: cpu, pickle_module=parlai.utils.pickle
            )

        new_classifier_head = False
        num_classes = len(self.class_list)

        if "model" in states:
            model_state_dict = states["model"]

            # first check if number of classes is different
            # this will happen when we're finetuning with a different class list
            if num_classes != model_state_dict["classifier_head.weight"].shape[0]:
                new_classifier_head = True
                # update classifier weights in model state dict and load
                model_state_dict[
                    "classifier_head.weight"
                ] = self.model.classifier_head.weight
                model_state_dict[
                    "classifier_head.bias"
                ] = self.model.classifier_head.bias
            # now load state dict
            self.load_state_dict(states["model"])

        if "optimizer" in states:
            if self.opt["init_new_optim"]:  # remove optimizer state dict
                states.pop("optimizer")
                states.pop("lr_scheduler")
                states.pop("optimizer_type")
                states.pop("lr_scheduler_type")
            else:
                if new_classifier_head:  # update optimizer
                    opt_state_dict = states["optimizer"]["state"]
                    saved_groups = states["optimizer"]["param_groups"]

                    optim_params = [
                        (n, p)
                        for n, p in self.model.named_parameters()
                        if p.requires_grad
                    ]
                    saved_optim_params = list(
                        chain.from_iterable((g["params"] for g in saved_groups))
                    )
                    assert len(optim_params) == len(saved_optim_params)
                    assert len(list(self.model.named_parameters())) == len(
                        list(self.model.parameters())
                    )  # if different sizes, this method will probably break
                    id_map = {
                        old_id: n_p
                        for old_id, n_p in zip(saved_optim_params, optim_params)
                    }

                    # see https://pytorch.org/docs/stable/_modules/torch/optim/adam.html#Adam
                    updated_opt_states = dict()
                    model = (
                        self.model.module
                        if hasattr(self.model, "module")
                        else self.model
                    )
                    for k, v in opt_state_dict.items():
                        if k in id_map:
                            name, param = id_map[k]
                            if name == "classifier_head.weight":  # re-initialize
                                for inner_k in v.keys():
                                    if inner_k == "step":
                                        continue
                                    v[inner_k] = torch.zeros_like(
                                        model.classifier_head.weight,
                                        memory_format=torch.preserve_format,
                                    ).float()
                            elif name == "classifier_head.bias":  # re-initialize
                                for inner_k in v.keys():
                                    if inner_k == "step":
                                        continue
                                    v[inner_k] = torch.zeros_like(
                                        model.classifier_head.bias,
                                        memory_format=torch.preserve_format,
                                    ).float()

                            updated_opt_states[k] = v
                    states["optimizer"]["state"] = updated_opt_states

            if hasattr(self, "optimizer"):
                self.optimizer.load_state_dict(states["optimizer"])
        return states
