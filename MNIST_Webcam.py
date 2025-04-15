import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import random
from torchvision import datasets, transforms
from tqdm import tqdm, trange
import cv2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"using:{device}")

class mnist_cnn(nn.Module):
    def __init__(self):
        super().__init__()
        # First block: 32 3x3 filters
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        # Second block: 64 3x3 filters
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        # Calculate the flattened size after two 2x2 max pools (28/2/2 = 7)
        self.fc1 = nn.Linear(7 * 7 * 64, 256)
        self.fc2 = nn.Linear(256, 10)

        self.dropout = nn.Dropout(0.12)

    def forward(self, x):
        # First block
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)
        
        # Second block
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)
        
        # Flatten and fully connected layers
        x = x.view(-1, 7 * 7 * 64)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x) 
        #x = self.dropout(x) # Fixed: now actually calling the layer
        
        return x
    
mnist_train = datasets.MNIST(root="./datasets", train=True, transform=transforms.ToTensor(), download=True)
mnist_test = datasets.MNIST(root="./datasets", train=False, transform=transforms.ToTensor(), download=True)
train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=100, shuffle=True)
test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=100, shuffle=False)
model = mnist_cnn().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in trange(5):
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    
    for images, labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device)
        
        # Zero out the gradients
        optimizer.zero_grad()

        # Forward pass
        y = model(images)
        loss = criterion(y, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Update progress bar with loss info
        progress_bar.set_description(f"Training (loss={loss.item():.4f})")

correct = 0
total = len(mnist_test)

with torch.no_grad():
    print("Testing...")
    progress_bar = tqdm(test_loader, desc="Testing", position=0, leave=True)
    
    # Iterate through test set minibatchs
    for images, labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        y = model(images)

        predictions = torch.argmax(y, dim=1)
        batch_correct = torch.sum((predictions == labels).float()).item()
        correct += batch_correct
        
        # Update progress bar with batch accuracy
        progress_bar.set_description(f"Testing (acc={batch_correct/len(labels):.4f})")

print(f'Test accuracy: {(correct*100/total):.2f}%')

#press q to close
# 1.creating a video object
video = cv2.VideoCapture(0) 
# 2. Variable
a = 0
prediction = 0
# 3. While loop
while True:
    a = a + 1
    # 4.Create a frame object
    check, frame = video.read()
    # get the size of the webcam video
    height, width = frame.shape[:2]
    # calculate size of square and centre it
    size = min(height, width)
    start_x = (width - size) // 2
    start_y = (height - size) // 2
    square_frame = frame[start_y:start_y+size, start_x:start_x+size]
    # Converting to grayscale
    gray = cv2.cvtColor(square_frame,cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    gray = 255 - gray
    gamma = 0.5  # < 1 makes darks darker, lights pop
    lookup_table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype("uint8")
    gray = cv2.LUT(gray, lookup_table)
    processed_display = cv2.resize(gray, (150, 150))
    cv2.imshow("Processed", processed_display)
    display_frame = frame.copy()
    cv2.rectangle(display_frame, 
                  (start_x, start_y), 
                  (start_x + size, start_y + size), 
                  (0, 255, 0), 2)
    cv2.putText(display_frame, f"Predicted: {prediction}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    # 5.show the frame!
    cv2.imshow("Original with square outline", display_frame)
    # 6.for playing 
    key = cv2.waitKey(1)

    with torch.no_grad():
        resized = cv2.resize(gray, (28,28))
        mnist_tensor = torch.from_numpy(resized).float() / 255.0
        mnist_tensor = (mnist_tensor - 0.1307) / 0.3081
        mnist_tensor = mnist_tensor.unsqueeze(0).unsqueeze(0)
        mnist_tensor = mnist_tensor.to(device)
        output = model(mnist_tensor)
        prediction = torch.argmax(output, dim=1).item()
    if key == ord('q'):
        # Save the current frame when q is pressed
        cv2.imwrite("mnist_capture.jpg", gray)
        cv2.imwrite("mnist_28x28.jpg", resized)
        print(f"Captured image saved. Prediction: {prediction}")
    if key == ord('e'):
        break

# 8. shutdown the camera
video.release()
cv2.destroyAllWindows()