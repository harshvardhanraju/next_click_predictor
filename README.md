# Next Click Predictor 🎯

An AI-powered system that predicts where users will click next on web interfaces using computer vision, Bayesian networks, and explainable AI.

[![Demo](https://img.shields.io/badge/Demo-Live-brightgreen)](https://your-vercel-app.vercel.app)
[![API](https://img.shields.io/badge/API-Railway-blue)](https://your-railway-app.railway.app/docs)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🚀 Features

- **Computer Vision**: Automatic UI element detection using OpenCV and EasyOCR
- **Bayesian Networks**: Probabilistic modeling for accurate predictions
- **Explainable AI**: Detailed explanations for every prediction
- **Real-time Processing**: Fast predictions in 1-2 seconds
- **Interactive Visualization**: Overlay predictions on uploaded screenshots
- **Open Source**: Fully open source with comprehensive documentation

## 🎮 Try It Live

**🌐 Web App**: [your-app.vercel.app](https://your-vercel-app.vercel.app)  
**📚 API Docs**: [your-api.railway.app/docs](https://your-railway-app.railway.app/docs)

## 🖼️ How It Works

1. **Upload Screenshot**: Drop a PNG/JPG of any web interface
2. **User Profile**: Specify age, tech level, device type, and browsing style  
3. **Task Description**: Describe what the user is trying to accomplish
4. **AI Analysis**: Computer vision detects UI elements, Bayesian network predicts clicks
5. **Explainable Results**: Get predictions with detailed reasoning and confidence scores

## 🧠 Technical Architecture

### Core Components
- **Screenshot Processor**: OpenCV + EasyOCR for UI element detection
- **Feature Integrator**: Combines user, task, and visual features  
- **Bayesian Network Engine**: Dynamic probabilistic modeling with pgmpy
- **Explanation Generator**: Factor analysis and reasoning chain construction
- **Web Service**: FastAPI REST API with comprehensive endpoints

### Tech Stack
- **Backend**: Python, FastAPI, OpenCV, EasyOCR, pgmpy, scikit-learn
- **Frontend**: Next.js, React, TypeScript, Tailwind CSS
- **Deployment**: Railway (backend), Vercel (frontend)
- **Containerization**: Docker with optimized image

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+
- Docker (optional)

### Backend Setup
```bash
# Clone repository
git clone https://github.com/harshvardhanraju/next_click_predictor.git
cd next_click_predictor

# Install dependencies
pip install -r requirements.txt

# Run API server
uvicorn src.web_service:app --reload --port 8000
```

### Frontend Setup
```bash
# Navigate to frontend
cd frontend

# Install dependencies
npm install

# Set environment variable
echo "NEXT_PUBLIC_API_URL=http://localhost:8000" > .env.local

# Run development server
npm run dev
```

Visit `http://localhost:3000` to use the app locally.

### Docker Setup (Optimized)
```bash
# Build optimized Docker image (83MB vs 6.87GB original)
docker build -t next-click-predictor:optimized .
docker run -p 8000:8000 next-click-predictor:optimized

# For production deployment
docker run -d --name next-click-predictor \
  --restart unless-stopped \
  -p 8000:8000 \
  next-click-predictor:optimized
```

> **🚀 Docker Optimization**: Our Docker image has been optimized from **6.87GB** to just **83MB** (98.8% reduction) while maintaining full functionality!

## 📖 Documentation

- **[System Architecture](SYSTEM_ARCHITECTURE.md)**: Detailed technical documentation
- **[Deployment Guide](DEPLOYMENT.md)**: Production deployment instructions  
- **[Technical Summary](TECHNICAL_SUMMARY.md)**: High-level system overview
- **[API Documentation](https://your-railway-app.railway.app/docs)**: Interactive API docs

## 🔮 Example Use Cases

### E-commerce Optimization
- Predict next clicks in checkout flows
- Optimize product page layouts
- A/B testing validation

### UX Research  
- Understand user behavior patterns
- Identify confusing interface elements
- Accessibility improvements

### Personalization
- Adaptive interfaces based on user profiles
- Context-aware recommendations
- Smart UI assistance

## 📊 Performance

- **Processing Time**: 1-2 seconds per prediction
- **Accuracy**: High confidence scores with uncertainty quantification
- **Scalability**: Stateless design supports horizontal scaling
- **Memory Usage**: ~300-500MB during processing
- **Docker Image Size**: Optimized to 83MB (98.8% smaller than original)

## 🧪 Example Prediction

```json
{
  "top_prediction": {
    "element_type": "BUTTON",
    "click_probability": 0.87,
    "text": "Submit Order",
    "reasoning": ["High visual prominence", "Strong task alignment", "User experience match"]
  },
  "explanation": {
    "main_explanation": "Submit button predicted with 87% confidence due to high visual prominence (35% weight) and direct task relevance (28% weight)",
    "key_factors": [
      {
        "factor": "Visual Prominence", 
        "weight": 0.35,
        "description": "Large, centrally positioned with high contrast"
      },
      {
        "factor": "Task Relevance",
        "weight": 0.28, 
        "description": "Button text 'Submit Order' aligns perfectly with checkout task"
      }
    ]
  }
}
```

## 🛠️ Development

### Project Structure
```
├── src/                    # Backend source code
│   ├── next_click_predictor.py    # Main orchestrator
│   ├── bayesian_network.py        # Probabilistic inference
│   ├── screenshot_processor.py    # Computer vision
│   ├── feature_integrator.py      # Feature engineering  
│   ├── explanation_generator.py   # Explainable AI
│   └── web_service.py             # FastAPI web service
├── frontend/               # Next.js frontend
├── tests/                  # Test suite
├── examples/               # Usage examples
├── Dockerfile             # Container configuration
└── requirements.txt       # Python dependencies
```

### Running Tests
```bash
# Backend tests
python -m pytest tests/

# Frontend tests  
cd frontend && npm test
```

### Contributing
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📈 Roadmap

- [ ] **Mobile Interface Support**: Enhanced mobile UI detection
- [ ] **Eye-tracking Integration**: Real-time gaze data for improved accuracy  
- [ ] **Deep Learning Models**: CNN-based element recognition
- [ ] **Reinforcement Learning**: Adaptive model improvement from user feedback
- [ ] **Multi-language Support**: Internationalization for global use
- [ ] **Browser Extension**: Direct integration with web browsers

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Ways to Contribute
- 🐛 Report bugs and issues
- 💡 Suggest new features
- 📝 Improve documentation
- 🧪 Add test cases
- 🎨 Enhance UI/UX
- ⚡ Performance optimizations

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenCV**: Computer vision capabilities
- **EasyOCR**: Text detection and recognition
- **pgmpy**: Bayesian network implementation
- **FastAPI**: Modern Python web framework
- **Next.js**: React framework for frontend
- **Railway & Vercel**: Deployment platforms

## 📞 Support

- **Documentation**: Check our comprehensive docs
- **Issues**: [GitHub Issues](https://github.com/harshvardhanraju/next_click_predictor/issues)
- **Discussions**: [GitHub Discussions](https://github.com/harshvardhanraju/next_click_predictor/discussions)

---

Built with ❤️ by [harshvardhanraju](https://github.com/harshvardhanraju)

**⭐ Star this repository if you find it useful!**