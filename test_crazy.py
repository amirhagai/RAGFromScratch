from airllm import AutoModel
import timeit


start = timeit.default_timer()
model = AutoModel.from_pretrained("unsloth/Meta-Llama-3.1-405B-Instruct-bnb-4bit")

input_text = ['can you explain me about the axiom of choice intuitively?',]

input_tokens = model.tokenizer(input_text,
      return_tensors="pt", 
      return_attention_mask=False, 
      truncation=True, 
      max_length=128, 
      padding=False)

generation_output = model.generate(
      input_tokens['input_ids'].cuda(), 
      max_new_tokens=50,
      return_dict_in_generate=True)

output = model.tokenizer.decode(generation_output.sequences[0])


end = timeit.default_timer()

print(output)
print(f'time taken - {end - start}')

for out in output:
      print(out)