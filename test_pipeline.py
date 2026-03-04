"""Quick test script to verify RAG pipeline setup"""
import sys
from pathlib import Path

# Add src to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    try:
        from config.config import load_config, RAGConfig
        from data.loader import EmailDataLoader
        from retrieval.indexer import FAISSIndexBuilder
        from retrieval.retriever import HybridRetriever
        from retrieval.reranker import CrossEncoderReranker
        from generation.rag_generator import RAGGenerator
        from app.gradio_app import create_demo
        print("✅ All imports successful!")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_config():
    """Test configuration loading"""
    print("\nTesting configuration...")
    try:
        from config.config import load_config
        config = load_config('hospital_base', project_root=PROJECT_ROOT)
        print(f"✅ Config loaded: {config.dataset.name}")
        print(f"   Data path: {config.dataset.data_path}")
        print(f"   Index path: {config.index_path}")
        return True
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        return False

def test_data_loader():
    """Test data loading"""
    print("\nTesting data loader...")
    try:
        from config.config import load_config
        from data.loader import EmailDataLoader

        config = load_config('hospital_base', project_root=PROJECT_ROOT)

        if not config.dataset.data_path.exists():
            print(f"⚠️  Data file not found: {config.dataset.data_path}")
            print("   Please copy data files first!")
            return False

        loader = EmailDataLoader(config)
        documents = loader.load_documents()
        print(f"✅ Loaded {len(documents)} documents")
        print(f"   First doc preview: {documents[0].content[:100]}...")
        return True
    except Exception as e:
        print(f"❌ Data loader test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("="*60)
    print("RAG Pipeline Test Suite")
    print("="*60)

    results = []
    results.append(("Imports", test_imports()))
    results.append(("Configuration", test_config()))
    results.append(("Data Loader", test_data_loader()))

    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)

    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")

    all_passed = all(r[1] for r in results)

    if all_passed:
        print("\n🎉 All tests passed! You're ready to run the launcher.")
        print("\nNext steps:")
        print("1. Open notebooks/launcher.ipynb")
        print("2. Set MODE='full' and DATASET='hospital'")
        print("3. Run all cells to build index and launch demo")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")

    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
