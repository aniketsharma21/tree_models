"""Demo script showing the hybrid configuration system usage.

This script demonstrates how to use the tree_model_helper configuration system
which combines Python type safety with YAML flexibility.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tree_model_helper.config import (
    load_config, XGBoostConfig, ConfigLoader, get_config_template,
    create_experiment_config, get_production_config, show_config_info,
    list_available_configs
)

def demo_basic_usage():
    """Demonstrate basic configuration usage."""
    print("🎯 BASIC CONFIGURATION USAGE")
    print("=" * 40)

    # 1. Load from YAML file
    print("1. Loading XGBoost config from YAML:")
    try:
        config = load_config('config/defaults/xgboost_default.yaml')
        print(f"   ✅ Loaded {config.model.model_type} config")
        print(f"   📊 N_estimators: {config.model.n_estimators}")
        print(f"   🎯 Target column: {config.data.target_col}")
        print(f"   🔧 Tuning enabled: {config.tuning.enable_tuning}")
    except Exception as e:
        print(f"   ❌ Error: {e}")

    print()

    # 2. Create from Python objects
    print("2. Creating config from Python objects:")
    model_config = XGBoostConfig(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        reg_alpha=0.1
    )
    print(f"   ✅ Created XGBoost config")
    print(f"   📈 Parameters: {model_config.to_dict()}")

    print()


def demo_advanced_features():
    """Demonstrate advanced configuration features."""
    print("🚀 ADVANCED CONFIGURATION FEATURES")
    print("=" * 40)

    # 1. Custom loader with validation
    print("1. Custom loader with validation:")
    try:
        loader = ConfigLoader(validate=True, allow_environment_override=True)
        config = loader.load_config('config/defaults/lightgbm_default.yaml')
        print(f"   ✅ Loaded and validated {config.model.model_type} config")
        print(f"   🔍 Feature selection: {config.feature_selection.enable_feature_selection}")
    except Exception as e:
        print(f"   ❌ Error: {e}")

    print()

    # 2. Environment overrides
    print("2. Environment variable overrides:")
    os.environ['MODEL_TYPE'] = 'lightgbm'
    os.environ['N_ESTIMATORS'] = '300'
    os.environ['ENABLE_TUNING'] = 'false'

    try:
        config = load_config('config/defaults/xgboost_default.yaml')  # Base config
        print(f"   ✅ Base config loaded with environment overrides")
        print(f"   🔧 Model type: {config.model.model_type}")
        print(f"   📊 N_estimators: {config.model.n_estimators}")
        print(f"   ⚡ Tuning: {config.tuning.enable_tuning}")
    except Exception as e:
        print(f"   ❌ Error: {e}")

    # Clean up environment
    for key in ['MODEL_TYPE', 'N_ESTIMATORS', 'ENABLE_TUNING']:
        os.environ.pop(key, None)

    print()

    # 3. Configuration templates
    print("3. Configuration template generation:")
    try:
        template = get_config_template('catboost')
        print(f"   ✅ Generated CatBoost template")
        print(f"   📝 Template length: {len(template)} characters")
        print("   📄 First few lines:")
        for line in template.split('\n')[:5]:
            print(f"      {line}")
    except Exception as e:
        print(f"   ❌ Error: {e}")

    print()


def demo_production_usage():
    """Demonstrate production configuration usage."""
    print("🏭 PRODUCTION CONFIGURATION")
    print("=" * 35)

    # 1. Production config
    print("1. Loading production configuration:")
    try:
        prod_config = get_production_config('xgboost')
        print(f"   ✅ Loaded production config")
        print(f"   🔧 Model: {prod_config.model.model_type}")
        print(f"   📊 N_estimators: {prod_config.model.n_estimators}")
        print(f"   🎯 Feature selection: {prod_config.feature_selection.enable_feature_selection}")
        print(f"   ⚡ Tuning: {prod_config.tuning.enable_tuning}")
        print(f"   📈 MLflow experiment: {prod_config.mlflow.experiment_name}")
    except Exception as e:
        print(f"   ❌ Error: {e}")

    print()

    # 2. Custom experiment config
    print("2. Creating custom experiment config:")
    try:
        exp_config = create_experiment_config(
            model_type='lightgbm',
            data_path='data/fraud_detection.csv',
            experiment_name='fraud_detection_experiment',
            enable_tuning=True,
            n_trials=50
        )
        print(f"   ✅ Created experiment config")
        print(f"   📂 Data path: {exp_config.data.train_path}")
        print(f"   🧪 Experiment: {exp_config.mlflow.experiment_name}")
        print(f"   🔧 Tuning trials: {exp_config.tuning.n_trials}")
    except Exception as e:
        print(f"   ❌ Error: {e}")

    print()


def demo_utilities():
    """Demonstrate configuration utilities."""
    print("🛠️  CONFIGURATION UTILITIES")
    print("=" * 35)

    # 1. List available configs
    print("1. Available configurations:")
    try:
        available_configs = list_available_configs()
        print(f"   ✅ Found {len(available_configs)} configurations:")
        for name, path in available_configs.items():
            print(f"      📄 {name}: {path.name}")
    except Exception as e:
        print(f"   ❌ Error: {e}")

    print()

    # 2. Configuration info
    print("2. Configuration summary:")
    try:
        config = load_config('config/defaults/xgboost_default.yaml')
        info = show_config_info(config)
        print("   ✅ Configuration summary generated:")
        for line in info.split('\n')[:15]:  # Show first 15 lines
            print(f"   {line}")
        print("   ... (truncated)")
    except Exception as e:
        print(f"   ❌ Error: {e}")

    print()


def demo_comparison():
    """Demonstrate config vs YAML comparison."""
    print("⚖️  CONFIG.PY vs YAML COMPARISON")
    print("=" * 40)

    print("✅ ADVANTAGES OF HYBRID APPROACH:")
    print("   🔒 Type Safety: Python objects with type hints")
    print("   🔧 Flexibility: YAML files for easy configuration")
    print("   🌍 Environment: Support for environment overrides") 
    print("   ✅ Validation: Built-in schema validation")
    print("   🚀 Performance: No parsing overhead for defaults")
    print("   📝 Documentation: Self-documenting with examples")
    print("   🔄 Compatibility: Backwards compatible with existing code")

    print()
    print("📋 USAGE RECOMMENDATIONS:")
    print("   🧪 Development: Use YAML files for experimentation")
    print("   🏭 Production: Use environment variables for secrets")
    print("   📦 Distribution: Ship with sensible Python defaults")
    print("   🎯 Teams: YAML configs for non-programmers")
    print("   🔧 Advanced: Python configs for complex logic")

    print()


def main():
    """Run all configuration demos."""
    print("🎯 TREE MODEL HELPER - HYBRID CONFIGURATION DEMO")
    print("=" * 60)
    print()

    demo_basic_usage()
    demo_advanced_features() 
    demo_production_usage()
    demo_utilities()
    demo_comparison()

    print("🎉 DEMO COMPLETED!")
    print("=" * 20)
    print()
    print("🚀 QUICK START:")
    print("   1. Use YAML files for experiments:")
    print("      config = load_config('config/xgboost_default.yaml')")
    print()
    print("   2. Use Python objects for programmatic access:")
    print("      model_config = XGBoostConfig(n_estimators=500)")
    print()
    print("   3. Override with environment variables:")
    print("      export MODEL_TYPE=lightgbm")
    print("      export N_ESTIMATORS=300")
    print()
    print("   4. Use production configs for deployment:")
    print("      config = get_production_config('xgboost')")


if __name__ == "__main__":
    main()
