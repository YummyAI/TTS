import torch

target = 7544 # or whatever target vocab size you want
dim = 0
fn = torch.randn # or torch.zeros

all_info = torch.load("origin_model.pth")
# print([i for i in all_info['model'].keys() if "gpt.text_embedding" in i])
embedding_weight = all_info['model']['gpt.text_embedding.weight']
head_weight = all_info['model']['gpt.text_head.weight']


# embedding_bias = all_info['model']['gpt.text_embedding.bias']
head_bias = all_info['model']['gpt.text_head.bias']

print(embedding_weight.shape)
print(head_weight.shape)

# print(embedding_weight.bias)
# print(head_weight.bias)
# weights['text_embedding.weight']

all_info['model']['gpt.text_embedding.weight'] = torch.stack(
    [ x for x in embedding_weight ] +
    [ fn(embedding_weight[0].shape).to(device=embedding_weight[0].device, dtype=embedding_weight[0].dtype) for _ in range(target - embedding_weight.shape[dim] ) ]
)

all_info['model']['gpt.text_head.weight'] = torch.stack(
    [ x for x in head_weight ] +
    [ fn(head_weight[0].shape ).to(device=head_weight[0].device, dtype=head_weight[0].dtype) for _ in range(target - head_weight.shape[dim] ) ]
)


# all_info['model']['gpt.text_embedding.bias'] = torch.concat(embedding_bias, fn(target - embeding_bias.shape[0]).to(device=embedding_bias.device, dtype=embedding_bias.dtype))

all_info['model']['gpt.text_head.bias'] = torch.concat((head_bias, fn(target - head_bias.shape[0]).to(device=head_bias.device, dtype=head_bias.dtype)), dim=0)

# print(all_info['model']['gpt.text_head.weight'].shape)

torch.save(all_info, "changed_model.pth")