"""Memoize unmodified upstream outputs for identical teacher-forced prefixes."""
import contextlib
import study_formats as formats
core=formats.core
BaseEngine=core.Engine


class FastEngine(BaseEngine):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self._upstream={};self._recipient_key=None;self._scoring=False

    def _prime(self,ids):
        key=tuple(ids)
        if key not in self._upstream:
            values={};handles=[]
            for layer,module in enumerate(self.model.model.layers):
                def capture(module,args,output,layer=layer):
                    values[layer]=output.detach().cpu()
                handles.append(module.register_forward_hook(capture))
            try:BaseEngine.forward(self,ids)
            finally:
                for handle in handles:handle.remove()
            self._upstream[key]=values
        return self._upstream[key]

    def cached_forward(self,ids,keep=1,spec=None,stats=None):
        if spec is None:return BaseEngine.forward(self,ids,keep,spec,stats)
        layer=spec[0]
        original_state=self._prime(ids)[layer].to('cuda')
        modules=list(self.model.model.layers[:layer+1]);originals=[m.forward for m in modules]
        try:
            for index,module in enumerate(modules):
                if index==layer:
                    module.forward=lambda *args,**kwargs:original_state
                else:
                    module.forward=lambda hidden_states,*args,**kwargs:hidden_states
            return BaseEngine.forward(self,ids,keep,spec,stats)
        finally:
            for module,forward in zip(modules,originals):module.forward=forward

    def forward(self,ids,keep=1,spec=None,stats=None):
        if not self._scoring:return BaseEngine.forward(self,ids,keep,spec,stats)
        if spec is None:
            self._prime(ids)
            return BaseEngine.forward(self,ids,keep,spec,stats)
        return self.cached_forward(ids,keep,spec,stats)

    def score_candidates(self,encoded,answers,spec=None):
        key=tuple(encoded['prompt_ids'])
        if key!=self._recipient_key:
            self._upstream.clear();self._recipient_key=key
        old=self._scoring;self._scoring=True
        try:return BaseEngine.score_candidates(self,encoded,answers,spec)
        finally:self._scoring=old
