# modules/iterative_processor.py
"""
Iterative Active Learning Process Module
Module for performing iterative learning combining YOLO models with Classification models
"""

import os
import glob
import time
import traceback
import pandas as pd
from datetime import datetime
from pathlib import Path
import shutil
import json
from tqdm import tqdm

# Local module imports
from modules.yolo_active_learning import YOLOActiveLearning

class IterativeProcessor:
    """
    Iterative Active Learning Process Manager
    Performs iterative learning utilizing Classification models for multiple YOLO models
    """
    
    def __init__(self, yolo_models_dir, classifier_path, image_dir, label_dir, output_dir,
                 conf_threshold=0.25, iou_threshold=0.5, class_conf_threshold=0.5, 
                 max_cycles=10, gpu_num=0):
        """
        Initialize Iterative Processor
        
        Args:
            yolo_models_dir (str): Directory containing YOLO models
            classifier_path (str): Classification model path
            image_dir (str): Image dataset path
            label_dir (str): Ground truth label path
            output_dir (str): Results save path
            conf_threshold (float): Object detection confidence threshold
            iou_threshold (float): IoU threshold
            class_conf_threshold (float): Classification model confidence threshold
            max_cycles (int): Maximum number of training iterations
            gpu_num (int): GPU number to use
        """
        self.yolo_models_dir = yolo_models_dir
        self.classifier_path = classifier_path
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.output_dir = output_dir
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.class_conf_threshold = class_conf_threshold
        self.max_cycles = max_cycles
        self.gpu_num = gpu_num
        
        # Print configuration information
        print("🔧 Iterative Processor Configuration:")
        print(f"  - YOLO model directory: {yolo_models_dir}")
        print(f"  - Classification model: {classifier_path}")
        print(f"  - Image directory: {image_dir}")
        print(f"  - Label directory: {label_dir}")
        print(f"  - Results save: {output_dir}")
        print(f"  - Maximum cycles: {max_cycles}")
        print(f"  - GPU: {gpu_num}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Validate models
        self._validate_inputs()
        
        print("✅ Iterative Processor initialization completed")
        
    def _validate_inputs(self):
        """Validate input parameters"""
        # Check YOLO model directory
        if not os.path.exists(self.yolo_models_dir):
            raise FileNotFoundError(f"YOLO model directory not found: {self.yolo_models_dir}")
            
        # Check classification model file
        if not os.path.exists(self.classifier_path):
            raise FileNotFoundError(f"Classification model not found: {self.classifier_path}")
            
        # Check image directory
        if not os.path.exists(self.image_dir):
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
            
        # Check label directory
        if not os.path.exists(self.label_dir):
            raise FileNotFoundError(f"Label directory not found: {self.label_dir}")
        
        print("✅ Input parameter validation completed")
    
    def get_yolo_models(self):
        """Get list of YOLO model files"""
        print("🔍 Searching for YOLO model files...")
        
        # Search for .pt files
        model_paths = glob.glob(os.path.join(self.yolo_models_dir, "*.pt"))
        
        if not model_paths:
            raise Exception(f"No YOLO models (.pt files) found: {self.yolo_models_dir}")
        
        # Sort by filename
        model_paths.sort()
        
        print(f"📊 Found {len(model_paths)} YOLO models:")
        for i, path in enumerate(model_paths):
            model_name = os.path.basename(path)
            file_size = os.path.getsize(path) / (1024 * 1024)  # MB
            print(f"  {i+1:2d}. {model_name} ({file_size:.1f} MB)")
        
        return model_paths
    
    def run_single_experiment(self, model_path):
        """
        Run Active Learning experiment for a single YOLO model
        
        Args:
            model_path (str): YOLO model file path
            
        Returns:
            dict: Experiment result information
        """
        model_filename = os.path.basename(model_path)
        model_name = os.path.splitext(model_filename)[0]
        
        # Model-specific output directory
        model_output_dir = os.path.join(self.output_dir, model_name)
        
        print(f"\n{'='*80}")
        print(f"🚀 Starting YOLO model experiment: {model_filename}")
        print(f"📁 Results save path: {model_output_dir}")
        print(f"{'='*80}")
        
        # Record experiment start time
        start_time = time.time()
        
        try:
            # Create YOLOActiveLearning instance
            print("🏗️ Initializing Active Learning system...")
            active_learning = YOLOActiveLearning(
                model_path=model_path,
                classifier_path=self.classifier_path,
                image_dir=self.image_dir,
                label_dir=self.label_dir,
                output_dir=model_output_dir,
                conf_threshold=self.conf_threshold,
                iou_threshold=self.iou_threshold,
                class_conf_threshold=self.class_conf_threshold,
                max_cycles=self.max_cycles,
                gpu_num=self.gpu_num,
                use_classifier=True  # Use classification model
            )
            
            print("✅ Active Learning system initialization completed")
            
            # Run Active Learning process
            print("🔄 Running Active Learning process...")
            active_learning.run()
            
            # Calculate execution time
            elapsed_time = time.time() - start_time
            
            print(f"✅ YOLO model {model_filename} experiment completed")
            print(f"⏰ Execution time: {elapsed_time/60:.1f} minutes")
            
            return {
                "status": "Completed",
                "message": "Successfully executed",
                "model_name": model_name,
                "model_path": model_path,
                "output_dir": model_output_dir,
                "elapsed_time": elapsed_time,
                "cycles_completed": self.max_cycles
            }
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            error_message = f"Error occurred: {str(e)}"
            error_detail = traceback.format_exc()
            
            print(f"\n{'!'*80}")
            print(f"❌ Error occurred: Error during {model_filename} experiment")
            print(f"💥 Error message: {str(e)}")
            print(f"⏰ Execution time: {elapsed_time/60:.1f} minutes")
            print(f"{'!'*80}")
            
            # Save error log
            self._save_error_log(model_output_dir, error_message, error_detail, elapsed_time)
            
            return {
                "status": "Failed",
                "message": str(e),
                "model_name": model_name,
                "model_path": model_path,
                "output_dir": model_output_dir,
                "elapsed_time": elapsed_time,
                "error_detail": error_detail
            }
    
    def _save_error_log(self, output_dir, error_message, error_detail, elapsed_time):
        """Save error log"""
        try:
            os.makedirs(output_dir, exist_ok=True)
            error_log_dir = os.path.join(output_dir, "error_logs")
            os.makedirs(error_log_dir, exist_ok=True)
            
            error_log_path = os.path.join(error_log_dir, "error.log")
            with open(error_log_path, "w", encoding='utf-8') as f:
                f.write(f"Experiment Execution Error Log\n")
                f.write(f"="*50 + "\n")
                f.write(f"Error occurrence time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Execution time: {elapsed_time/60:.1f} minutes\n")
                f.write(f"Error message: {error_message}\n\n")
                f.write(f"Detailed error content:\n")
                f.write(f"-"*50 + "\n")
                f.write(f"{error_detail}\n")
            
            print(f"📝 Error log saved: {error_log_path}")
            
        except Exception as log_error:
            print(f"⚠️ Error log save failed: {str(log_error)}")
    
    def collect_metrics(self, experiment_results):
        """
        Collect and integrate metrics from all experiment results
        
        Args:
            experiment_results (dict): Experiment results dictionary
            
        Returns:
            pandas.DataFrame: Integrated metrics dataframe
        """
        print("📊 Collecting experiment metrics...")
        
        combined_metrics_df = pd.DataFrame()
        collected_count = 0
        
        for model_name, result in experiment_results.items():
            if result["status"] == "Completed":
                metrics_file = os.path.join(result["output_dir"], "performance_metrics.csv")
                
                if os.path.exists(metrics_file):
                    try:
                        model_metrics = pd.read_csv(metrics_file)
                        
                        # Add model name if not present
                        if 'Model' not in model_metrics.columns:
                            model_metrics['Model'] = model_name
                        
                        # Add experiment information
                        model_metrics['Experiment_Status'] = 'Success'
                        model_metrics['Elapsed_Time'] = result.get("elapsed_time", 0)
                        
                        combined_metrics_df = pd.concat([combined_metrics_df, model_metrics], 
                                                       ignore_index=True)
                        collected_count += 1
                        
                        print(f"  ✅ {model_name}: Collected {len(model_metrics)} cycle metrics")
                        
                    except Exception as e:
                        print(f"  ❌ {model_name}: Metrics file read error - {str(e)}")
                else:
                    print(f"  ⚠️ {model_name}: No metrics file")
            else:
                print(f"  💥 {model_name}: No metrics due to experiment failure")
        
        print(f"📈 Metrics collection completed for {collected_count} models")
        
        return combined_metrics_df
    
    def create_comparison_tables(self, combined_metrics_df, output_dir):
        """Create performance comparison tables"""
        print("📊 Creating performance comparison tables...")
        
        if combined_metrics_df.empty:
            print("⚠️ Cannot create comparison tables due to no metric data")
            return
        
        try:
            # Create comparison tables for each metric
            metrics_to_compare = ['mAP50', 'Precision', 'Recall', 'F1-Score', 
                                'Detected_Objects', 'Filtered_Objects']
            
            for metric in metrics_to_compare:
                if metric in combined_metrics_df.columns:
                    try:
                        # Create pivot table based on cycle and model
                        pivot_df = combined_metrics_df.pivot_table(
                            index='Cycle', 
                            columns='Model', 
                            values=metric,
                            aggfunc='mean'
                        )
                        
                        # Save table
                        table_file = os.path.join(output_dir, f"{metric}_comparison_table.csv")
                        pivot_df.to_csv(table_file)
                        
                        print(f"  📄 {metric} comparison table saved: {table_file}")
                        
                        # Final cycle performance summary
                        if not pivot_df.empty:
                            final_cycle = pivot_df.index.max()
                            final_performance = pivot_df.loc[final_cycle].sort_values(ascending=False)
                            
                            print(f"  🏆 {metric} final performance (cycle {final_cycle}):")
                            for model, score in final_performance.head(3).items():
                                if not pd.isna(score):
                                    print(f"    {model}: {score:.4f}")
                        
                    except Exception as e:
                        print(f"  ❌ {metric} table creation failed: {str(e)}")
                else:
                    print(f"  ⚠️ {metric} column not found")
            
            # Overall metrics summary table
            summary_file = os.path.join(output_dir, "performance_summary.csv")
            combined_metrics_df.to_csv(summary_file, index=False)
            print(f"📋 Overall performance summary saved: {summary_file}")
            
        except Exception as e:
            print(f"❌ Error during comparison table creation: {str(e)}")
    
    def generate_experiment_report(self, experiment_results, combined_metrics_df, output_dir):
        """Generate comprehensive experiment results report"""
        print("📝 Generating experiment report...")
        
        report_file = os.path.join(output_dir, "experiment_report.txt")
        
        try:
            with open(report_file, "w", encoding='utf-8') as f:
                # Report header
                f.write("="*80 + "\n")
                f.write("ITERATIVE ACTIVE LEARNING EXPERIMENT REPORT\n")
                f.write("="*80 + "\n")
                f.write(f"Generation time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Experiment settings:\n")
                f.write(f"  - Maximum cycles: {self.max_cycles}\n")
                f.write(f"  - Confidence threshold: {self.conf_threshold}\n")
                f.write(f"  - IoU threshold: {self.iou_threshold}\n")
                f.write(f"  - Classification confidence threshold: {self.class_conf_threshold}\n")
                f.write(f"  - GPU: {self.gpu_num}\n")
                f.write("\n")
                
                # Experiment results summary
                total_models = len(experiment_results)
                successful_count = sum(1 for r in experiment_results.values() if r["status"] == "Completed")
                failed_count = total_models - successful_count
                
                f.write("📊 Experiment Results Summary\n")
                f.write("-"*50 + "\n")
                f.write(f"Total models: {total_models}\n")
                f.write(f"Successful: {successful_count} ({successful_count/total_models*100:.1f}%)\n")
                f.write(f"Failed: {failed_count} ({failed_count/total_models*100:.1f}%)\n")
                f.write("\n")
                
                # Detailed results for each model
                f.write("📋 Detailed Results by Model\n")
                f.write("-"*50 + "\n")
                
                for model_name, result in experiment_results.items():
                    status = result["status"]
                    elapsed_time = result.get("elapsed_time", 0)
                    
                    f.write(f"\n🔹 {model_name}\n")
                    f.write(f"   Status: {status}\n")
                    f.write(f"   Execution time: {elapsed_time/60:.1f} minutes\n")
                    
                    if status == "Completed":
                        f.write(f"   Results directory: {result['output_dir']}\n")
                        f.write(f"   Completed cycles: {result.get('cycles_completed', 'N/A')}\n")
                    else:
                        f.write(f"   Error message: {result['message']}\n")
                
                # Performance statistics (if metrics available)
                if not combined_metrics_df.empty:
                    f.write("\n\n📈 Performance Statistics\n")
                    f.write("-"*50 + "\n")
                    
                    # Average performance of final cycle
                    final_cycle = combined_metrics_df['Cycle'].max()
                    final_metrics = combined_metrics_df[combined_metrics_df['Cycle'] == final_cycle]
                    
                    if not final_metrics.empty:
                        f.write(f"Final cycle ({final_cycle}) average performance:\n")
                        
                        for metric in ['mAP50', 'Precision', 'Recall', 'F1-Score']:
                            if metric in final_metrics.columns:
                                avg_score = final_metrics[metric].mean()
                                f.write(f"  {metric}: {avg_score:.4f}\n")
                        
                        f.write(f"\nObject detection statistics:\n")
                        if 'Detected_Objects' in final_metrics.columns:
                            total_detected = final_metrics['Detected_Objects'].sum()
                            f.write(f"  Total detected objects: {total_detected:,}\n")
                        
                        if 'Filtered_Objects' in final_metrics.columns:
                            total_filtered = final_metrics['Filtered_Objects'].sum()
                            f.write(f"  Filtered objects: {total_filtered:,}\n")
                            
                            if total_detected > 0:
                                filter_rate = total_filtered / (total_detected + total_filtered) * 100
                                f.write(f"  Filter rate: {filter_rate:.1f}%\n")
                    
                    # Best performing model
                    if 'F1-Score' in combined_metrics_df.columns:
                        best_performance = combined_metrics_df.loc[combined_metrics_df['F1-Score'].idxmax()]
                        f.write(f"\n🏆 Best performing model:\n")
                        f.write(f"  Model: {best_performance['Model']}\n")
                        f.write(f"  Cycle: {best_performance['Cycle']}\n")
                        f.write(f"  F1-Score: {best_performance['F1-Score']:.4f}\n")
                
                # List of experiment files
                f.write(f"\n\n📁 Generated Files\n")
                f.write("-"*50 + "\n")
                f.write(f"- Experiment report: {report_file}\n")
                f.write(f"- Performance summary: {os.path.join(output_dir, 'performance_summary.csv')}\n")
                f.write(f"- Comparison tables: {output_dir}/*_comparison_table.csv\n")
                f.write(f"- Model-specific results: {output_dir}/[model_name]/\n")
                
            print(f"📄 Experiment report saved: {report_file}")
            
        except Exception as e:
            print(f"❌ Report generation failed: {str(e)}")
    
    def save_experiment_config(self, output_dir):
        """Save experiment configuration to JSON file"""
        config = {
            "experiment_datetime": datetime.now().isoformat(),
            "yolo_models_dir": self.yolo_models_dir,
            "classifier_path": self.classifier_path,
            "image_dir": self.image_dir,
            "label_dir": self.label_dir,
            "output_dir": self.output_dir,
            "parameters": {
                "conf_threshold": self.conf_threshold,
                "iou_threshold": self.iou_threshold,
                "class_conf_threshold": self.class_conf_threshold,
                "max_cycles": self.max_cycles,
                "gpu_num": self.gpu_num
            }
        }
        
        config_file = os.path.join(output_dir, "experiment_config.json")
        try:
            with open(config_file, "w") as f:
                json.dump(config, f, indent=2)
            print(f"⚙️ Experiment configuration saved: {config_file}")
        except Exception as e:
            print(f"⚠️ Configuration save failed: {str(e)}")
    
    def run_iterative_experiments(self):
        """
        Run iterative experiments for all YOLO models
        
        Returns:
            dict: Comprehensive experiment results
        """
        print("="*80)
        print("🚀 STARTING ITERATIVE ACTIVE LEARNING EXPERIMENTS")
        print("="*80)
        
        # Record total experiment start time
        total_start_time = time.time()
        
        # Create experiment directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_dir = os.path.join(self.output_dir, f"iterative_experiment_{timestamp}")
        os.makedirs(experiment_dir, exist_ok=True)
        
        print(f"📁 Experiment results save directory: {experiment_dir}")
        
        # Save experiment configuration
        self.save_experiment_config(experiment_dir)
        
        # Get YOLO model list
        try:
            model_paths = self.get_yolo_models()
        except Exception as e:
            print(f"❌ YOLO model loading failed: {str(e)}")
            return {"error": str(e)}
        
        # Dictionary to store experiment results
        experiment_results = {}
        
        # Progress display with tqdm
        print(f"\n🔄 Starting experiments for {len(model_paths)} models")
        
        model_pbar = tqdm(model_paths, desc="Models", unit="model")
        
        # Perform experiments for each YOLO model
        for i, model_path in enumerate(model_pbar):
            model_filename = os.path.basename(model_path)
            model_name = os.path.splitext(model_filename)[0]
            
            # Update progress
            model_pbar.set_description(f"Processing {model_name}")
            
            print(f"\n🎯 Experiment progress: {i+1}/{len(model_paths)} - {model_name}")
            
            # Run individual model experiment
            result = self.run_single_experiment(model_path)
            experiment_results[model_name] = result
            
            # Output intermediate results
            if result["status"] == "Completed":
                print(f"✅ {model_name} completed ({result['elapsed_time']/60:.1f} minutes)")
            else:
                print(f"❌ {model_name} failed: {result['message']}")
        
        model_pbar.close()
        
        # Calculate total experiment time
        total_elapsed_time = time.time() - total_start_time
        
        print(f"\n⏰ Total experiment time: {total_elapsed_time/60:.1f} minutes")
        
        # Collect and integrate metrics
        print("\n📊 Analyzing experiment results...")
        combined_metrics_df = self.collect_metrics(experiment_results)
        
        # Save integrated metrics
        if not combined_metrics_df.empty:
            combined_metrics_file = os.path.join(experiment_dir, "combined_performance_metrics.csv")
            combined_metrics_df.to_csv(combined_metrics_file, index=False)
            print(f"💾 Integrated metrics saved: {combined_metrics_file}")
            
            # Create comparison tables
            self.create_comparison_tables(combined_metrics_df, experiment_dir)
        
        # Generate comprehensive report
        self.generate_experiment_report(experiment_results, combined_metrics_df, experiment_dir)
        
        # Final results summary
        successful_count = sum(1 for r in experiment_results.values() if r["status"] == "Completed")
        failed_count = len(experiment_results) - successful_count
        
        print("\n" + "="*80)
        print("🎉 ITERATIVE ACTIVE LEARNING EXPERIMENTS COMPLETED")
        print("="*80)
        print(f"📊 Experiment results:")
        print(f"  - Total models: {len(model_paths)}")
        print(f"  - Successful: {successful_count}")
        print(f"  - Failed: {failed_count}")
        print(f"  - Success rate: {successful_count/len(model_paths)*100:.1f}%")
        print(f"⏰ Total execution time: {total_elapsed_time/60:.1f} minutes")
        print(f"📁 Results save location: {experiment_dir}")
        
        if not combined_metrics_df.empty:
            print(f"📈 Metrics summary:")
            print(f"  - Integrated metrics: {len(combined_metrics_df)} records")
            print(f"  - Comparison tables: {len([f for f in os.listdir(experiment_dir) if f.endswith('_comparison_table.csv')])}")
        
        # Construct return results dictionary
        final_results = {
            "experiment_dir": experiment_dir,
            "results": experiment_results,
            "combined_metrics": combined_metrics_df,
            "summary": {
                "total_models": len(model_paths),
                "successful": successful_count,
                "failed": failed_count,
                "success_rate": successful_count/len(model_paths)*100,
                "total_time_minutes": total_elapsed_time/60,
                "experiments_per_hour": len(model_paths) / (total_elapsed_time / 3600)
            }
        }
        
        return final_results
    
    def cleanup_temp_files(self):
        """Clean up temporary files"""
        print("🧹 Cleaning up temporary files...")
        
        temp_patterns = [
            "runs/detect/*/",  # YOLO training temporary results
            "*.yaml",          # Temporary YAML files
            "__pycache__/",    # Python cache
        ]
        
        for pattern in temp_patterns:
            try:
                temp_files = glob.glob(pattern)
                for temp_file in temp_files:
                    if os.path.isdir(temp_file):
                        shutil.rmtree(temp_file, ignore_errors=True)
                    else:
                        os.remove(temp_file)
                
                if temp_files:
                    print(f"  🗑️ {len(temp_files)} {pattern} files/folders cleaned up")
            except Exception as e:
                print(f"  ⚠️ {pattern} cleanup failed: {str(e)}")
        
        print("✅ Temporary files cleanup completed")

def main():
    """Main function for test execution"""
    # Test configuration
    processor = IterativeProcessor(
        yolo_models_dir="./models/initial_yolo",
        classifier_path="./models/classification/densenet121_100.pth",
        image_dir="./dataset/images",
        label_dir="./dataset/labels",
        output_dir="./results/iterative_process",
        conf_threshold=0.25,
        iou_threshold=0.5,
        class_conf_threshold=0.5,
        max_cycles=5,  # Reduced for testing
        gpu_num=0
    )
    
    # Run experiments
    results = processor.run_iterative_experiments()
    
    # Output results
    if "error" not in results:
        print("\n🎯 Experiments completed! Key results:")
        summary = results["summary"]
        print(f"  - Success rate: {summary['success_rate']:.1f}%")
        print(f"  - Total time: {summary['total_time_minutes']:.1f} minutes")
        print(f"  - Experiments per hour: {summary['experiments_per_hour']:.1f}")
    else:
        print(f"❌ Experiments failed: {results['error']}")

if __name__ == "__main__":
    main()