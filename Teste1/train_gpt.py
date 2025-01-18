import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel, pipeline
from datasets import load_dataset

# 1. Carregar o tokenizador GPT-2
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Configurar o pad_token para o tokenizador
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 2. Definir o modelo personalizado baseado no GPT-2
config = GPT2Config(
    vocab_size=50257,  # Tamanho do vocabulário
    n_layer=6,         # Número de camadas do Transformer
    n_head=8,          # Número de cabeças de atenção
    n_embd=512         # Dimensão dos embeddings
)

class GPTCustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = GPT2LMHeadModel(config)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs.loss, outputs.logits

# 3. Preparar os dados
print("Carregando o conjunto de dados...")
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")["train"]

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Converter os dados para tensores
def convert_to_tensors(batch):
    return {
        "input_ids": torch.tensor(batch["input_ids"]),
        "attention_mask": torch.tensor(batch["attention_mask"])
    }

tokenized_dataset = tokenized_dataset.map(convert_to_tensors, batched=True)
dataloader = DataLoader(tokenized_dataset, batch_size=8, shuffle=True)

# 4. Instanciar o modelo e otimizador
print("Configurando o modelo e o treinamento...")
model = GPTCustomModel()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

optimizer = AdamW(model.parameters(), lr=5e-5)

# 5. Treinar o modelo
print("Iniciando o treinamento...")
num_epochs = 3
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    for step, batch in enumerate(dataloader):
        input_ids = torch.stack(batch["input_ids"]).to(device)
        attention_mask = torch.stack(batch["attention_mask"]).to(device)

        # Forward pass
        labels = input_ids.clone()  # Cópia para os labels
        loss, _ = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Imprimir o Loss para cada batch
        if step % 10 == 0:  # Imprimir a cada 10 passos para reduzir a quantidade de prints
            print(f"Epoch {epoch + 1}/{num_epochs}, Batch {step}/{len(dataloader)} - Loss: {loss.item():.4f}")

print("Treinamento concluído!")


# 6. Gerar texto com o modelo treinado
print("Testando geração de texto...")
generator = pipeline("text-generation", model=model.model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

# Geração de texto
output = generator("Era uma vez em um mundo distante,", max_length=50, num_return_sequences=1)
print("Texto gerado:")
print(output)
