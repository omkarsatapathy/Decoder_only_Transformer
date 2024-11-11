import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tokenizer import prepare_sentiment_data
from utils import Encoder
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from tqdm import tqdm
import argparse
from pathlib import Path
from config import Config
import logging

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Sentiment Analysis Training')
    parser.add_argument('--config', type=str, default='config.json',
                        help='Path to configuration file')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to data file')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs')
    return parser.parse_args()


def setup_training(config: Config):
    """Setup training components based on configuration"""

    # Set random seeds for reproducibility
    torch.manual_seed(config.data.random_seed)
    np.random.seed(config.data.random_seed)

    # Prepare data
    logger.info("Preparing sentiment data...")
    X, Y, tokenizer = prepare_sentiment_data(config)

    # Split data
    x_train, x_test, y_train, y_test = train_test_split(
        X, Y,
        test_size=(1 - config.data.train_size),
        random_state=config.data.random_seed
    )

    # Move data to device
    x_train = x_train.to(config.device)
    y_train = y_train.to(config.device)
    x_test = x_test.to(config.device)
    y_test = y_test.to(config.device)

    logger.info(f"Train set size: {len(x_train)}, Test set size: {len(x_test)}")

    # Create data loaders
    train_dataset = TensorDataset(x_train, y_train)
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        drop_last=True,
        # num_workers=config.training.num_workers,
        # pin_memory=config.training.pin_memory
    )

    val_dataset = TensorDataset(x_test, y_test)
    val_data_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        drop_last=True,
        # num_workers=config.training.num_workers,
        # pin_memory=config.training.pin_memory
    )

    # Initialize model
    logger.info("Initializing model...")
    model = Encoder(config).to(config.device)

    # Initialize optimizer and criterion
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.training.learning_rate)

    # Initialize scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config.training.scheduler_factor,
        patience=config.training.scheduler_patience,
        verbose=True
    )

    return model, criterion, optimizer, scheduler, train_data_loader, val_data_loader


def train_epoch(model, train_loader, criterion, optimizer, config):
    """Train for one epoch"""
    model.train()
    total_train_correct = 0
    total_train_samples = 0
    total_train_loss = 0

    progress_bar = tqdm(train_loader, desc='Training')
    for inputs_batch, y_batch in progress_bar:
        outputs = model(inputs_batch)
        loss = criterion(outputs, y_batch.view(-1))

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.clip_value)
        optimizer.step()

        # Update metrics
        total_train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total_train_correct += (predicted == y_batch).sum().item()
        total_train_samples += y_batch.size(0)

        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.0 * total_train_correct / total_train_samples:.2f}%'
        })

    return total_train_loss / len(train_loader), total_train_correct / total_train_samples


def validate(model, val_loader, criterion):
    """Validate the model"""
    model.eval()
    total_val_correct = 0
    total_val_samples = 0
    total_val_loss = 0
    all_val_preds = []
    all_val_labels = []

    with torch.no_grad():
        for val_inputs_batch, y_batch in val_loader:
            val_outputs = model(val_inputs_batch)
            val_loss = criterion(val_outputs, y_batch.view(-1))
            total_val_loss += val_loss.item()

            _, val_predicted = torch.max(val_outputs, 1)
            total_val_correct += (val_predicted == y_batch).sum().item()
            total_val_samples += y_batch.size(0)

            all_val_preds.extend(val_predicted.cpu().numpy())
            all_val_labels.extend(y_batch.cpu().numpy())

    return (
        total_val_loss / len(val_loader),
        total_val_correct / total_val_samples,
        all_val_preds,
        all_val_labels
    )


def main():
    # Parse command line arguments
    args = parse_args()

    # Load configuration
    config = Config.load(args.config) if Path(args.config).exists() else Config()

    # Override config with command line arguments
    if args.data_path:
        config.data.data_path = Path(args.data_path)
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.epochs:
        config.training.epochs = args.epochs

    # Display configuration
    logger.info("Starting training with configuration:")
    config.display()

    # Create output directories if they don't exist
    Path('checkpoints').mkdir(exist_ok=True)
    Path('plots').mkdir(exist_ok=True)

    # Setup training components
    model, criterion, optimizer, scheduler, train_loader, val_loader = setup_training(config)

    # Training loop
    logger.info("Starting training...")
    losses, val_losses = [], []
    best_val_loss = float('inf')

    for epoch in range(config.training.epochs):
        logger.info(f'\nEpoch {epoch + 1}/{config.training.epochs}:')

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, config)

        # Validate
        val_loss, val_acc, val_preds, val_labels = validate(model, val_loader, criterion)

        # Update learning rate
        scheduler.step(val_loss)

        # Save metrics
        losses.append(train_loss)
        val_losses.append(val_loss)

        # Print metrics
        logger.info(f'Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc * 100:.2f}%')
        logger.info(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc * 100:.2f}%')

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            logger.info(f'Saving best model with validation loss: {val_loss:.4f}')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': val_loss,
            }, 'checkpoints/best_model.pt')

    # Plot training history
    plot_training_history(losses, val_losses)

    # Plot confusion matrix
    plot_confusion_matrix(val_labels, val_preds)

    logger.info("Training completed!")


def plot_training_history(losses, val_losses):
    """Plot training and validation loss"""
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('plots/training_history.png')
    plt.close()


def plot_confusion_matrix(val_labels, val_preds):
    """Plot confusion matrix"""
    cm = confusion_matrix(val_labels, val_preds)
    plt.figure(figsize=(8, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.savefig('plots/confusion_matrix.png')
    plt.close()


if __name__ == "__main__":
    main()