from parlai.core.context import BaseContext


class PersonaContext(BaseContext):
    def __init__(self, persona_list):
        self.persona_list = persona_list

    @staticmethod
    def from_data(input_data):
        '''Persona context data is just an array of context strings, and can
        be initialized directly on them.
        '''
        return PersonaContext(input_data)

    def to_data_form(self):
        '''Returns the array of persona strings that represents this object'''
        return self.persona_list

    def to_context_string(self):
        '''Return the persona lines to be fed to a model'''
        return '\n'.join(
            ['Your persona: {}'.format(p) for p in self.persona_list]
        )

    def get_type(self):
        return 'persona'
