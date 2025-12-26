# Code Explanation - Client App Chi Tiết

Giải thích từng dòng code trong `pytorchexample/client_app.py`

---

## 📋 Tổng quan

File này định nghĩa **Client Application** trong Federated Learning:
- Nhận model từ server
- Train trên dữ liệu local
- Gửi model đã train về server
- Evaluate model trên validation set local

---

## 📦 Imports (Dòng 1-9)

### Dòng 3: Import PyTorch
```python
import torch
```
**Mục đích**: Framework deep learning chính
- Tạo tensors
- Quản lý GPU/CPU
- Load/save models

### Dòng 4: Import Flower Components
```python
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
```

**Chi tiết**:
- **`ArrayRecord`**: Container cho model weights (parameters)
  - Ví dụ: `ArrayRecord(model.state_dict())`
  - Chuyển PyTorch state_dict thành format Flower có thể truyền đi

- **`Context`**: Chứa thông tin về môi trường
  - `context.node_config`: Config của client này (ID, partition, etc.)
  - `context.run_config`: Config chung (learning rate, batch size, etc.)

- **`Message`**: Đối tượng truyền thông tin giữa client-server
  - `msg.content`: Nội dung (model weights, configs)
  - `msg.reply_to`: Trả lời cho message nào

- **`MetricRecord`**: Container cho metrics
  - Ví dụ: `{"loss": 1.5, "accuracy": 0.85}`

- **`RecordDict`**: Dictionary chứa các records
  - Ví dụ: `{"arrays": model_weights, "metrics": metrics_dict}`

### Dòng 5: Import ClientApp
```python
from flwr.clientapp import ClientApp
```
**Mục đích**: Class chính để tạo Flower client application
- Sử dụng decorators (`@app.train()`, `@app.evaluate()`)
- Tự động kết nối với server
- Quản lý communication protocol

### Dòng 7-9: Import task functions
```python
from pytorchexample.task import Net, load_data
from pytorchexample.task import test as test_fn
from pytorchexample.task import train as train_fn
```

**Chi tiết**:
- **`Net`**: Class định nghĩa CNN model
- **`load_data`**: Function load dữ liệu CIFAR-10 cho client
- **`test_fn`**: Function test model (đổi tên thành `test_fn` tránh conflict)
- **`train_fn`**: Function train model local

---

## 🏗️ Khởi tạo ClientApp (Dòng 11-12)

```python
# Flower ClientApp
app = ClientApp()
```

**Giải thích**:
- Tạo instance của `ClientApp`
- Object này sẽ được Flower sử dụng để:
  - Kết nối với server
  - Nhận messages (train/evaluate commands)
  - Gọi các functions được decorate

**Flow**:
```
Server → Message → ClientApp → @app.train() hoặc @app.evaluate()
```

---

## 🎓 Function Train (Dòng 15-48)

### Decorator (Dòng 15)
```python
@app.train()
def train(msg: Message, context: Context):
```

**Giải thích**:
- **`@app.train()`**: Decorator đánh dấu function này xử lý training
- **`msg`**: Message từ server chứa:
  - Global model weights
  - Training config (learning rate)
- **`context`**: Thông tin môi trường:
  - Client ID
  - Partition ID
  - Run configs

### Bước 1: Load Model & Weights (Dòng 19-23)

```python
# Load the model and initialize it with the received weights
model = Net()
model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
```

**Chi tiết từng dòng**:

**Dòng 20**: `model = Net()`
- Tạo CNN model mới (architecture rỗng, chưa có weights)

**Dòng 21**: `model.load_state_dict(msg.content["arrays"].to_torch_state_dict())`
- `msg.content["arrays"]`: Lấy model weights từ message
- `.to_torch_state_dict()`: Chuyển từ Flower format → PyTorch format
- `model.load_state_dict(...)`: Load weights vào model

**Ví dụ state_dict**:
```python
{
  'conv1.weight': tensor([[...]]),
  'conv1.bias': tensor([...]),
  'fc1.weight': tensor([[...]]),
  ...
}
```

**Dòng 22**: `device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")`
- Check GPU có sẵn không
- `cuda:0`: GPU đầu tiên
- `cpu`: Fallback nếu không có GPU

**Dòng 23**: `model.to(device)`
- Move model lên GPU (nếu có) hoặc giữ ở CPU

### Bước 2: Load Data (Dòng 25-29)

```python
# Load the data
partition_id = context.node_config["partition-id"]
num_partitions = context.node_config["num-partitions"]
batch_size = context.run_config["batch-size"]
trainloader, _ = load_data(partition_id, num_partitions, batch_size)
```

**Chi tiết**:

**Dòng 26**: `partition_id = context.node_config["partition-id"]`
- Lấy ID của client này
- Ví dụ: Client 0, Client 1, ..., Client 9

**Dòng 27**: `num_partitions = context.node_config["num-partitions"]`
- Tổng số clients
- Ví dụ: 10 clients → num_partitions = 10

**Dòng 28**: `batch_size = context.run_config["batch-size"]`
- Batch size cho training
- Ví dụ: 32

**Dòng 29**: `trainloader, _ = load_data(partition_id, num_partitions, batch_size)`
- `load_data()`: Load dữ liệu CIFAR-10 partition cho client này
- `trainloader`: DataLoader cho training
- `_`: Ignore testloader (không dùng ở đây)

**Ví dụ partition**:
```
Total data: 50,000 images
10 clients → Mỗi client: 5,000 images

Client 0: images 0-4,999
Client 1: images 5,000-9,999
Client 2: images 10,000-14,999
...
```

### Bước 3: Train Model (Dòng 31-38)

```python
# Call the training function
train_loss = train_fn(
    model,
    trainloader,
    context.run_config["local-epochs"],
    msg.content["config"]["lr"],
    device,
)
```

**Chi tiết**:

**Parameters**:
- `model`: Model với global weights
- `trainloader`: Dữ liệu local của client
- `context.run_config["local-epochs"]`: Số epochs train local (ví dụ: 1)
- `msg.content["config"]["lr"]`: Learning rate (ví dụ: 0.1)
- `device`: GPU hoặc CPU

**Process bên trong `train_fn()`**:
```python
# Pseudo code
for epoch in range(local_epochs):  # Ví dụ: 1 epoch
    for batch in trainloader:
        images, labels = batch
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
```

**Return**: `train_loss` (average loss sau khi train)

### Bước 4: Gửi về Server (Dòng 40-48)

```python
# Construct and return reply Message
model_record = ArrayRecord(model.state_dict())
metrics = {
    "train_loss": train_loss,
    "num-examples": len(trainloader.dataset),
}
metric_record = MetricRecord(metrics)
content = RecordDict({"arrays": model_record, "metrics": metric_record})
return Message(content=content, reply_to=msg)
```

**Chi tiết từng dòng**:

**Dòng 41**: `model_record = ArrayRecord(model.state_dict())`
- Lấy weights của model sau khi train
- Chuyển thành Flower format

**Dòng 42-45**: Tạo metrics dictionary
```python
metrics = {
    "train_loss": train_loss,      # Loss sau training (ví dụ: 1.234)
    "num-examples": len(trainloader.dataset),  # Số samples (ví dụ: 4000)
}
```

**Dòng 46**: `metric_record = MetricRecord(metrics)`
- Wrap metrics vào Flower format

**Dòng 47**: `content = RecordDict(...)`
- Tạo dictionary chứa:
  - `"arrays"`: Model weights đã train
  - `"metrics"`: Training metrics

**Dòng 48**: `return Message(content=content, reply_to=msg)`
- Tạo message reply
- `reply_to=msg`: Liên kết với message request ban đầu
- Gửi về server

**Flow hoàn chỉnh**:
```
Server sends:
  - Global model weights
  - Config (lr, epochs)
    ↓
Client receives
    ↓
Client trains local
    ↓
Client sends back:
  - Updated weights
  - Training loss
  - Number of examples
```

---

## 📊 Function Evaluate (Dòng 51-82)

### Decorator (Dòng 51-52)
```python
@app.evaluate()
def evaluate(msg: Message, context: Context):
```

**Giải thích**:
- Tương tự `@app.train()` nhưng cho evaluation
- Server yêu cầu client evaluate model hiện tại

### Bước 1: Load Model (Dòng 55-59)

```python
# Load the model and initialize it with the received weights
model = Net()
model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
```

**Giống với train**, nhưng:
- Weights có thể là aggregated weights từ nhiều clients
- Không train, chỉ evaluate

### Bước 2: Load Validation Data (Dòng 61-65)

```python
# Load the data
partition_id = context.node_config["partition-id"]
num_partitions = context.node_config["num-partitions"]
batch_size = context.run_config["batch-size"]
_, valloader = load_data(partition_id, num_partitions, batch_size)
```

**Khác biệt**:
- Dòng 65: `_, valloader` - Lấy **validation loader** (không phải trainloader)
- Mỗi client có validation set riêng (20% của data local)

**Ví dụ split**:
```
Client 0 có 5000 images:
  - Training: 4000 images (80%)
  - Validation: 1000 images (20%)
```

### Bước 3: Evaluate (Dòng 67-72)

```python
# Call the evaluation function
eval_loss, eval_acc = test_fn(
    model,
    valloader,
    device,
)
```

**Process bên trong `test_fn()`**:
```python
# Pseudo code
model.eval()  # Set to evaluation mode
with torch.no_grad():  # Không tính gradients
    for batch in valloader:
        images, labels = batch
        outputs = model(images)
        loss = criterion(outputs, labels)
        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum()

return average_loss, accuracy
```

**Return**:
- `eval_loss`: Average loss trên validation set
- `eval_acc`: Accuracy (số dự đoán đúng / tổng số)

### Bước 4: Gửi Metrics về Server (Dòng 74-82)

```python
# Construct and return reply Message
metrics = {
    "eval_loss": eval_loss,
    "eval_acc": eval_acc,
    "num-examples": len(valloader.dataset),
}
metric_record = MetricRecord(metrics)
content = RecordDict({"metrics": metric_record})
return Message(content=content, reply_to=msg)
```

**Chi tiết**:

**Dòng 75-79**: Tạo metrics
```python
{
    "eval_loss": 1.234,      # Loss trên validation set
    "eval_acc": 0.753,       # Accuracy (75.3%)
    "num-examples": 1000,    # Số validation samples
}
```

**Khác với train**:
- **KHÔNG** gửi model weights về
- Chỉ gửi metrics để server monitor

**Dòng 81**: `content = RecordDict({"metrics": metric_record})`
- Chỉ có `"metrics"`, không có `"arrays"`

---

## 🔄 Complete Workflow

### Round 1 của Federated Learning:

#### Step 1: Server → Client (Train Request)
```
Server sends Message:
  msg.content["arrays"] = Global model weights (random init)
  msg.content["config"]["lr"] = 0.1
```

#### Step 2: Client Training
```python
@app.train() được gọi:
  1. Load global weights vào model local
  2. Load data partition của client
  3. Train 1 epoch trên data local
  4. Return updated weights + metrics
```

#### Step 3: Client → Server (Train Response)
```
Client sends Message:
  content["arrays"] = Updated weights
  content["metrics"] = {
    "train_loss": 1.234,
    "num-examples": 4000
  }
```

#### Step 4: Server Aggregates
```
Server nhận weights từ tất cả clients
Server tính average (FedAvg):
  new_weights = average([client1_weights, client2_weights, ...])
```

#### Step 5: Server → Client (Evaluate Request)
```
Server sends Message:
  msg.content["arrays"] = Aggregated weights
```

#### Step 6: Client Evaluation
```python
@app.evaluate() được gọi:
  1. Load aggregated weights
  2. Load validation data
  3. Evaluate (no training)
  4. Return metrics only
```

#### Step 7: Client → Server (Evaluate Response)
```
Client sends Message:
  content["metrics"] = {
    "eval_loss": 1.189,
    "eval_acc": 0.567,
    "num-examples": 1000
  }
```

#### Step 8: Round Complete
```
Server nhận metrics từ tất cả clients
Server log global metrics
Round 2 begins...
```

---

## 📊 Data Flow Diagram

```
┌─────────────────────────────────────────────────────┐
│                    SERVER                           │
│  - Global model                                     │
│  - Aggregation strategy (FedAvg)                    │
└────────┬────────────────────────────────────┬───────┘
         │                                    │
    Train Request                        Evaluate Request
    (global weights)                     (aggregated weights)
         │                                    │
         ▼                                    ▼
┌─────────────────────────────────────────────────────┐
│                   CLIENT                            │
│                                                     │
│  @app.train():                @app.evaluate():      │
│  1. Load weights              1. Load weights       │
│  2. Load data (80%)           2. Load data (20%)    │
│  3. Train local               3. Evaluate only      │
│  4. Return weights            4. Return metrics     │
└────────┬────────────────────────────────────┬───────┘
         │                                    │
    Train Response                       Evaluate Response
    (updated weights + metrics)          (metrics only)
         │                                    │
         ▼                                    ▼
┌─────────────────────────────────────────────────────┐
│                    SERVER                           │
│  - Aggregate weights                                │
│  - Compute global metrics                           │
│  - Next round                                       │
└─────────────────────────────────────────────────────┘
```

---

## 💡 Key Concepts

### 1. Local Training
- Mỗi client train trên **data riêng**
- Không share raw data
- Chỉ share **model weights**

### 2. Data Partition
```python
partition_id = 0  # Client 0
num_partitions = 10  # Total 10 clients

Client 0: 10% của dataset (5000 images)
Client 1: 10% khác (5000 images)
...
```

### 3. Train vs Evaluate

| Aspect | Train | Evaluate |
|--------|-------|----------|
| **Data** | Training set (80%) | Validation set (20%) |
| **Process** | Update weights | No update |
| **Return** | Weights + metrics | Metrics only |
| **Gradients** | Yes | No |

### 4. Message Communication

**Train Message**:
```python
Request:  arrays (global weights) + config (lr)
Response: arrays (updated weights) + metrics (loss, num_examples)
```

**Evaluate Message**:
```python
Request:  arrays (aggregated weights)
Response: metrics (loss, accuracy, num_examples)
```

---

## 🎯 Summary

### Client App làm gì?

1. **Nhận** global model từ server
2. **Train** trên dữ liệu local
3. **Gửi** updated weights về server
4. **Evaluate** aggregated model
5. **Report** metrics về server

### Tại sao quan trọng?

- ✅ **Privacy**: Không share raw data
- ✅ **Distributed**: Train parallel trên nhiều clients
- ✅ **Scalable**: Thêm client dễ dàng
- ✅ **Flexible**: Mỗi client có thể có data khác nhau

### Next Steps

Để hiểu đầy đủ hơn, đọc thêm:
- `server_app.py` - Server side logic
- `task.py` - Model architecture & training
- `strategies.py` - Aggregation algorithms

---

**Code này là core của Federated Learning! 🚀**
