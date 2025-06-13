# Script para extraer parámetros del scaler y eliminar dependencia de joblib
import joblib
import json
import os

def extraer_parametros_scaler():
    """Extrae parámetros del scaler MinMaxScaler y los guarda en JSON"""
    try:
        print("📦 Cargando scaler desde scalerMinMax.pkl...")
        scaler = joblib.load("modelo/scalerMinMax.pkl")
        
        # Extraer parámetros
        scaler_params = {
            "min_": scaler.min_.tolist(),
            "scale_": scaler.scale_.tolist(),
            "data_min_": scaler.data_min_.tolist() if hasattr(scaler, 'data_min_') else None,
            "data_max_": scaler.data_max_.tolist() if hasattr(scaler, 'data_max_') else None,
            "data_range_": scaler.data_range_.tolist() if hasattr(scaler, 'data_range_') else None,
            "feature_range": scaler.feature_range,
            "n_features_in_": int(scaler.n_features_in_) if hasattr(scaler, 'n_features_in_') else None
        }
        
        print("💾 Guardando parámetros en scaler_params.json...")
        with open("modelo/scaler_params.json", "w") as f:
            json.dump(scaler_params, f, indent=2)
        
        print("✅ Parámetros del scaler extraídos exitosamente")
        
        # Mostrar información
        print(f"📊 Información del scaler:")
        print(f"   Feature range: {scaler_params['feature_range']}")
        print(f"   Número de features: {scaler_params['n_features_in_']}")
        print(f"   Min values: {scaler_params['min_']}")
        print(f"   Scale values: {scaler_params['scale_']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error extrayendo parámetros: {e}")
        return False

def verificar_scaler_ligero():
    """Verifica que el scaler ligero funcione igual que el original"""
    try:
        print("\n🔍 Verificando compatibilidad del scaler...")
        
        # Cargar scaler original
        scaler_original = joblib.load("modelo/scalerMinMax.pkl")
        
        # Importar y cargar scaler ligero
        import sys
        sys.path.append('.')
        
        class ScalerLigero:
            def __init__(self, archivo_params):
                with open(archivo_params, 'r') as f:
                    params = json.load(f)
                self.min_ = params["min_"]
                self.scale_ = params["scale_"]
            
            def transform(self, X):
                import numpy as np
                X = np.array(X)
                min_ = np.array(self.min_)
                scale_ = np.array(self.scale_)
                return (X - min_) * scale_
        
        scaler_ligero = ScalerLigero("modelo/scaler_params.json")
        
        # Datos de prueba
        import numpy as np
        datos_prueba = np.array([[50000, 75000], [25000, 45000], [100000, 120000]])
        
        # Transformar con ambos scalers
        result_original = scaler_original.transform(datos_prueba)
        result_ligero = scaler_ligero.transform(datos_prueba)
        
        # Comparar resultados
        diferencia = np.abs(result_original - result_ligero).max()
        
        print(f"   Diferencia máxima: {diferencia}")
        
        if diferencia < 1e-10:
            print("✅ El scaler ligero funciona correctamente")
            return True
        else:
            print("⚠️ Hay diferencias en los resultados")
            print(f"Original: {result_original[0]}")
            print(f"Ligero: {result_ligero[0]}")
            return False
            
    except Exception as e:
        print(f"❌ Error verificando scaler: {e}")
        return False

def mostrar_tamaños_archivos():
    """Muestra los tamaños de los archivos del modelo"""
    print("\n📏 Tamaños de archivos:")
    
    archivos = [
        "modelo/bestModel.keras",
        "modelo/scalerMinMax.pkl", 
        "modelo/model.onnx",
        "modelo/scaler_params.json"
    ]
    
    total_original = 0
    total_optimizado = 0
    
    for archivo in archivos:
        if os.path.exists(archivo):
            size_bytes = os.path.getsize(archivo)
            size_kb = size_bytes / 1024
            size_mb = size_bytes / (1024 * 1024)
            
            if size_mb > 1:
                print(f"   {archivo}: {size_mb:.1f} MB")
            else:
                print(f"   {archivo}: {size_kb:.1f} KB")
            
            if archivo.endswith(('.keras', 'scalerMinMax.pkl')):
                total_original += size_bytes
            elif archivo.endswith(('.onnx', 'scaler_params.json')):
                total_optimizado += size_bytes
    
    if total_original > 0 and total_optimizado > 0:
        reduction = (1 - total_optimizado / total_original) * 100
        print(f"\n🎯 Reducción de tamaño: {reduction:.1f}%")
        print(f"   Original: {total_original / (1024*1024):.1f} MB")
        print(f"   Optimizado: {total_optimizado / (1024*1024):.1f} MB")

if __name__ == "__main__":
    # Crear directorio si no existe
    os.makedirs("modelo", exist_ok=True)
    
    print("🚀 Extrayendo parámetros del scaler...")
    
    if extraer_parametros_scaler():
        if verificar_scaler_ligero():
            print("\n🎉 ¡Optimización completada exitosamente!")
            print("\n📋 Próximos pasos:")
            print("1. Actualiza requirements.txt:")
            print("   flask")
            print("   onnxruntime")
            print("   numpy")
            print("\n2. Los archivos .keras y .pkl ya no son necesarios")
            print("3. Usa solo model.onnx y scaler_params.json")
            
            mostrar_tamaños_archivos()
        else:
            print("⚠️ Revisa la extracción del scaler")
    else:
        print("❌ No se pudieron extraer los parámetros")