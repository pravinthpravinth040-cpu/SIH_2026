import io
from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms

from model import get_resnet18_classifier

_model = None
_device = None
_transform = None

def get_transform(img_size=224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def load_inference_model(checkpoint_path=None):
    global _model, _device, _transform
    if _model is not None:
        return _model, _device, _transform
        
    if checkpoint_path is None:
        checkpoint_path = Path(__file__).parent / "checkpoints" / "best_model.pth"
    else:
        checkpoint_path = Path(checkpoint_path)
        
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at: {checkpoint_path}")
        
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _model = get_resnet18_classifier(pretrained=False)
    
    checkpoint = torch.load(checkpoint_path, map_location=_device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        _model.load_state_dict(checkpoint['model_state_dict'])
    else:
        _model.load_state_dict(checkpoint)
        
    _model = _model.to(_device)
    _model.eval()
    _transform = get_transform()
    return _model, _device, _transform

def predict(image_input, checkpoint_path=None):
    """
    Predicts whether an input SAR image contains an oil spill.
    
    Args:
        image_input: File path (str/Path), image bytes, or PIL Image instance.
        checkpoint_path: Path to .pth weights file (optional).
        
    Returns:
        dict: {
            'oil_detected': bool,
            'confidence': float,
            'raw_score': float
        }
    """
    model, device, transform = load_inference_model(checkpoint_path)
    
    if isinstance(image_input, (str, Path)):
        img = Image.open(image_input).convert('RGB')
    elif isinstance(image_input, bytes):
        img = Image.open(io.BytesIO(image_input)).convert('RGB')
    elif isinstance(image_input, Image.Image):
        img = image_input.convert('RGB')
    else:
        raise ValueError("Unsupported image input type. Pass a file path, bytes, or PIL Image.")
        
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(img_tensor).squeeze(1)
        prob = torch.sigmoid(output).item()
        
    oil_detected = prob >= 0.5
    confidence = prob if oil_detected else (1.0 - prob)
    
    return {
        'oil_detected': bool(oil_detected),
        'confidence': float(confidence),
        'raw_score': float(prob)
    }

if __name__ == "__main__":
    sample_path = Path(__file__).parent / "sample_images" / "oil_sample_1.jpg"
    if sample_path.exists():
        res = predict(sample_path)
        print(f"Sample prediction: Oil Detected = {res['oil_detected']}, Confidence = {res['confidence']:.4f}")
