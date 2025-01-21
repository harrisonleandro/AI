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
    batch["input_ids"] = [torch.tensor(ids, dtype=torch.long) for ids in batch["input_ids"]]
    batch["attention_mask"] = [torch.tensor(mask, dtype=torch.long) for mask in batch["attention_mask"]]
    return batch

tokenized_dataset = tokenized_dataset.map(convert_to_tensors, batched=True)

# Criar uma função de colagem para o DataLoader
def collate_fn(batch):
    input_ids = torch.stack([torch.tensor(item["input_ids"]) for item in batch])
    attention_mask = torch.stack([torch.tensor(item["attention_mask"]) for item in batch])
    return {"input_ids": input_ids, "attention_mask": attention_mask}

dataloader = DataLoader(tokenized_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

# 4. Instanciar o modelo e otimizador
print("Configurando o modelo e o treinamento...")
model = GPTCustomModel()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

optimizer = AdamW(model.parameters(), lr=5e-5)

# 5. Treinar o modelo
print("Iniciando o treinamento...")
num_epochs = 5  # Número de épocas
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    model.train()  # Modo de treinamento
    for step, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        # Forward pass
        labels = input_ids.clone()  # Cópia para os labels
        loss, _ = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Imprimir o Loss para cada batch
        if step % 10 == 0:  # Imprimir a cada 10 passos
            print(f"Epoch {epoch + 1}/{num_epochs}, Batch {step}/{len(dataloader)} - Loss: {loss.item():.4f}")

print("Treinamento concluído!")

# 6. Gerar texto com o modelo treinado
print("Testando geração de texto...")
generator = pipeline("text-generation", model=model.model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

# Geração de texto
output = generator(
    "Era uma vez em um mundo distante,", 
    max_length=100,  # Permitir textos mais longos
    num_return_sequences=3,  # Gerar múltiplas saídas
    temperature=0.8,  # Controlar aleatoriedade
    top_k=50,  # Limitar aos 50 tokens mais prováveis
    top_p=0.9,  # Usar amostragem de núcleo
    truncation=True  # Ativar truncamento explicitamente
)

print("Texto(s) gerado(s):")
for idx, text in enumerate(output):
    print(f"Saída {idx + 1}: {text['generated_text']}")
