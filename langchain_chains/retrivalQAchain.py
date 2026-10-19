class Prompt:
    def __init__(self, template, input_variable):
        self.template = template
        self.input_variable = input_variable

    def format(self, input_dict):
        print(self.template)
        return self.template.format(**input_dict)

llm = Prompt(
    template="write a story {topics}.",
    input_variable=['topic']
)

print(llm.format({'topics': 'village'}))
