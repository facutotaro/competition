import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import base64

# Set page configuration
st.set_page_config(
    page_title="MNIST Digit Generator",
    page_icon="🎯",
    layout="centered"
)

# Generator Network (same as training script)
class Generator(nn.Module):
    def __init__(self, z_dim, num_classes, img_size, channels):
        super(Generator, self).__init__()
        self.img_size = img_size
        self.channels = channels
        
        # Embedding for class labels
        self.label_embedding = nn.Embedding(num_classes, num_classes)
        
        # Generator layers
        self.gen = nn.Sequential(
            nn.Linear(z_dim + num_classes, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, img_size * img_size * channels),
            nn.Tanh()
        )
    
    def forward(self, noise, labels):
        label_embedding = self.label_embedding(labels)
        gen_input = torch.cat([noise, label_embedding], dim=1)
        img = self.gen(gen_input)
        img = img.view(img.size(0), self.channels, self.img_size, self.img_size)
        return img

# Mock image generation function (replace with actual model loading)
def generate_mock_images(digit, num_images=5):
    """Generate mock images for demonstration"""
    images = []
    
    for i in range(num_images):
        # Create a 28x28 image
        fig, ax = plt.subplots(figsize=(1, 1), dpi=28)
        ax.text(0.5, 0.5, str(digit), fontsize=20, ha='center', va='center', 
                transform=ax.transAxes, weight='bold')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        # Add some randomization
        noise = np.random.normal(0, 0.1, (28, 28))
        ax.imshow(noise, alpha=0.3, cmap='gray')
        
        # Convert to image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        img = Image.open(buf)
        images.append(img)
        plt.close()
    
    return images

# Function to load the trained model
@st.cache_resource
def load_model():
    """Load the trained GAN model"""
    try:
        checkpoint = torch.load('mnist_gan_model.pth', map_location='cpu')
        model = Generator(
            z_dim=checkpoint['z_dim'],
            num_classes=checkpoint['num_classes'], 
            img_size=checkpoint['img_size'],
            channels=checkpoint['channels']
        )
        model.load_state_dict(checkpoint['generator_state_dict'])
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Function to generate images using the trained model
def generate_digit_images(digit, num_images=5, model=None):
    """Generate images for a specific digit"""
    if model is None:
        # Use mock generation for demonstration
        return generate_mock_images(digit, num_images)
    
    # Real model generation code
    device = torch.device('cpu')
    z_dim = 100
    
    with torch.no_grad():
        noise = torch.randn(num_images, z_dim).to(device)
        labels = torch.full((num_images,), digit, dtype=torch.long).to(device)
        generated_imgs = model(noise, labels)
        
        # Convert to PIL images
        generated_imgs = generated_imgs.cpu().numpy()
        generated_imgs = (generated_imgs + 1) / 2.0  # Denormalize from [-1,1] to [0,1]
        generated_imgs = np.clip(generated_imgs, 0, 1)
        
        images = []
        for img in generated_imgs:
            img_array = (img.squeeze() * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_array, mode='L')
            # Resize for better display (28x28 is quite small)
            pil_img = pil_img.resize((112, 112), Image.NEAREST)
            images.append(pil_img)
        
        return images

# Main Streamlit app
def main():
    st.title("🎯 MNIST Digit Generator")
    st.markdown("### Generate handwritten-style digits using a trained GAN model")
    
    # Load model
    model = load_model()
    
    if model is not None:
        st.success("✅ GAN model loaded successfully!")
    else:
        st.warning("⚠️ Using demo mode. Make sure 'mnist_gan_model.pth' is in your repository.")
    
    # Digit selection
    st.markdown("#### Select a digit to generate (0-9)")
    
    # Create columns for digit buttons
    cols = st.columns(10)
    selected_digit = None
    
    for i, col in enumerate(cols):
        if col.button(str(i), key=f"digit_{i}", use_container_width=True):
            selected_digit = i
            st.session_state['selected_digit'] = i
    
    # Check if digit was selected previously
    if 'selected_digit' in st.session_state:
        selected_digit = st.session_state['selected_digit']
    
    if selected_digit is not None:
        st.success(f"Selected digit: **{selected_digit}**")
        
        # Generate button
        if st.button("🎨 Generate 5 Images", type="primary", use_container_width=True):
            with st.spinner(f"Generating images for digit {selected_digit}..."):
                try:
                    # Generate images
                    images = generate_digit_images(selected_digit, 5, model)
                    
                    # Display images
                    st.markdown(f"#### Generated Images for Digit {selected_digit}")
                    
                    # Create columns for images
                    img_cols = st.columns(5)
                    
                    for i, (col, img) in enumerate(zip(img_cols, images)):
                        with col:
                            st.image(img, caption=f"Image {i+1}", use_container_width=True)
                    
                    # Display additional info
                    st.markdown("---")
                    st.markdown("""
                    **Model Information:**
                    - **Architecture:** Conditional GAN (Generator + Discriminator)
                    - **Dataset:** MNIST (28x28 grayscale images)
                    - **Training:** From scratch using PyTorch
                    - **Framework:** PyTorch with Adam optimizer
                    - **Loss Function:** Binary Cross Entropy
                    """)
                    
                except Exception as e:
                    st.error(f"Error generating images: {e}")
    
    else:
        st.info("👆 Please select a digit above to generate images")
    
    # Sidebar with additional information
    with st.sidebar:
        st.markdown("## About This App")
        st.markdown("""
        This web application generates handwritten digit images using a 
        Generative Adversarial Network (GAN) trained on the MNIST dataset.
        
        **Features:**
        - Select any digit (0-9)
        - Generate 5 unique images
        - MNIST-style 28x28 grayscale format
        
        **Technical Details:**
        - **Model:** Conditional GAN
        - **Training:** Google Colab (T4 GPU)
        - **Framework:** PyTorch
        - **Dataset:** MNIST
        """)
        
        st.markdown("---")
        st.markdown("**How it works:**")
        st.markdown("""
        1. The Generator creates fake digit images
        2. The Discriminator tries to detect fake images
        3. Both networks improve through adversarial training
        4. Result: Realistic handwritten digits
        """)

if __name__ == "__main__":
    main()