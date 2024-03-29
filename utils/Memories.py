
class ExtendedConversationBufferMemory(ConversationBufferMemory):
    
    extra_variables:List[str] = []

    @property
    def memory_variables(self) -> List[str]:
        return [self.memory_key] + self.extra_variables

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        d = super().load_memory_variables(inputs)
        d.update({k:inputs.get(k) for k in self.extra_variables})        
        return d