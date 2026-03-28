#!/bin/bash

echo "🚀 PFedRec Project Setup"
echo "========================"

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please install Docker Desktop first."
    exit 1
fi

# Check Docker Compose
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose not found."
    exit 1
fi

# Create data directory
mkdir -p data/ml-100k

echo "✅ Prerequisites checked"
echo ""
echo "📥 Next Steps:"
echo "1. Download MovieLens-100K: https://grouplens.org/datasets/movielens/100k/"
echo "2. Extract u.data to data/ml-100k/u.data"
echo "3. Run: python data/prepare_data.py"
echo "4. Run: docker-compose up --build"
echo ""
echo "🌐 Server Dashboard: http://localhost:5000/status"
echo "📊 Ray Dashboard: http://localhost:8265 (if using distributed profile)"