SC2_INSTRUCT_PROMPT = """You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions.

### Instruction
{instruction}

### Response
{response}"""

CHAT_TEMPLATE = """{{bos_token}}{{'You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions.\n\n'}}
{%- for message in messages %}
    {%- if message['role'] == 'system' %}
        {{ raise_exception('System messages are not allowed in this template.') }}
    {%- else %}
        {%- if message['role'] == 'user' %}
{{'### Instruction\n' + message['content'] + '\n\n'}}
        {%- else %}
{{'### Response\n' + message['content'] + eos_token + '\n\n'}}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{{'### Response\n'}}"""