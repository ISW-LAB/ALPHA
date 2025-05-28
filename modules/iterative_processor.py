# modules/iterative_processor.py
"""
Iterative Active Learning Process 모듈
여러 YOLO 모델에 대해 Classification 모델과 결합된 반복적 학습을 수행하는 모듈
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

# 로컬 모듈 임포트
from modules.yolo_active_learning import YOLOActiveLearning

class IterativeProcessor:
    """
    Iterative Active Learning 프로세스 관리자
    여러 YOLO 모델에 대해 Classification 모델을 활용한 반복적 학습을 수행
    """
    
    def __init__(self, yolo_models_dir, classifier_path, image_dir, label_dir, output_dir,
                 conf_threshold=0.25, iou_threshold=0.5, class_conf_threshold=0.5, 
                 max_cycles=10, gpu_num=0):
        """
        Iterative Processor 초기화
        
        Args:
            yolo_models_dir (str): YOLO 모델들이 저장된 디렉토리
            classifier_path (str): 분류 모델 경로
            image_dir (str): 이미지 데이터셋 경로
            label_dir (str): 정답 라벨 경로
            output_dir (str): 결과 저장 경로
            conf_threshold (float): 객체 검출 신뢰도 임계값
            iou_threshold (float): IoU 임계값
            class_conf_threshold (float): 분류 모델 신뢰도 임계값
            max_cycles (int): 최대 학습 반복 횟수
            gpu_num (int): 사용할 GPU 번호
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
        
        # 설정 정보 출력
        print("🔧 Iterative Processor 설정:")
        print(f"  - YOLO 모델 디렉토리: {yolo_models_dir}")
        print(f"  - 분류 모델: {classifier_path}")
        print(f"  - 이미지 디렉토리: {image_dir}")
        print(f"  - 라벨 디렉토리: {label_dir}")
        print(f"  - 결과 저장: {output_dir}")
        print(f"  - 최대 사이클: {max_cycles}")
        print(f"  - GPU: {gpu_num}")
        
        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        
        # 모델 검증
        self._validate_inputs()
        
        print("✅ Iterative Processor 초기화 완료")
        
    def _validate_inputs(self):
        """입력 파라미터 검증"""
        # YOLO 모델 디렉토리 확인
        if not os.path.exists(self.yolo_models_dir):
            raise FileNotFoundError(f"YOLO 모델 디렉토리를 찾을 수 없습니다: {self.yolo_models_dir}")
            
        # 분류 모델 파일 확인
        if not os.path.exists(self.classifier_path):
            raise FileNotFoundError(f"분류 모델을 찾을 수 없습니다: {self.classifier_path}")
            
        # 이미지 디렉토리 확인
        if not os.path.exists(self.image_dir):
            raise FileNotFoundError(f"이미지 디렉토리를 찾을 수 없습니다: {self.image_dir}")
            
        # 라벨 디렉토리 확인
        if not os.path.exists(self.label_dir):
            raise FileNotFoundError(f"라벨 디렉토리를 찾을 수 없습니다: {self.label_dir}")
        
        print("✅ 입력 파라미터 검증 완료")
    
    def get_yolo_models(self):
        """YOLO 모델 파일 목록 가져오기"""
        print("🔍 YOLO 모델 파일 검색 중...")
        
        # .pt 파일 검색
        model_paths = glob.glob(os.path.join(self.yolo_models_dir, "*.pt"))
        
        if not model_paths:
            raise Exception(f"YOLO 모델(.pt 파일)을 찾을 수 없습니다: {self.yolo_models_dir}")
        
        # 파일명 기준으로 정렬
        model_paths.sort()
        
        print(f"📊 총 {len(model_paths)}개의 YOLO 모델을 찾았습니다:")
        for i, path in enumerate(model_paths):
            model_name = os.path.basename(path)
            file_size = os.path.getsize(path) / (1024 * 1024)  # MB
            print(f"  {i+1:2d}. {model_name} ({file_size:.1f} MB)")
        
        return model_paths
    
    def run_single_experiment(self, model_path):
        """
        단일 YOLO 모델에 대한 Active Learning 실험 실행
        
        Args:
            model_path (str): YOLO 모델 파일 경로
            
        Returns:
            dict: 실험 결과 정보
        """
        model_filename = os.path.basename(model_path)
        model_name = os.path.splitext(model_filename)[0]
        
        # 모델별 출력 디렉토리
        model_output_dir = os.path.join(self.output_dir, model_name)
        
        print(f"\n{'='*80}")
        print(f"🚀 YOLO 모델 실험 시작: {model_filename}")
        print(f"📁 결과 저장 경로: {model_output_dir}")
        print(f"{'='*80}")
        
        # 실험 시작 시간 기록
        start_time = time.time()
        
        try:
            # YOLOActiveLearning 인스턴스 생성
            print("🏗️ Active Learning 시스템 초기화 중...")
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
                use_classifier=True  # 분류 모델 사용
            )
            
            print("✅ Active Learning 시스템 초기화 완료")
            
            # Active Learning 프로세스 실행
            print("🔄 Active Learning 프로세스 실행 중...")
            active_learning.run()
            
            # 실행 시간 계산
            elapsed_time = time.time() - start_time
            
            print(f"✅ YOLO 모델 {model_filename} 실험 완료")
            print(f"⏰ 실행 시간: {elapsed_time/60:.1f}분")
            
            return {
                "status": "완료",
                "message": "성공적으로 실행됨",
                "model_name": model_name,
                "model_path": model_path,
                "output_dir": model_output_dir,
                "elapsed_time": elapsed_time,
                "cycles_completed": self.max_cycles
            }
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            error_message = f"오류 발생: {str(e)}"
            error_detail = traceback.format_exc()
            
            print(f"\n{'!'*80}")
            print(f"❌ 오류 발생: {model_filename} 실험 중 오류")
            print(f"💥 오류 메시지: {str(e)}")
            print(f"⏰ 실행 시간: {elapsed_time/60:.1f}분")
            print(f"{'!'*80}")
            
            # 오류 로그 저장
            self._save_error_log(model_output_dir, error_message, error_detail, elapsed_time)
            
            return {
                "status": "실패",
                "message": str(e),
                "model_name": model_name,
                "model_path": model_path,
                "output_dir": model_output_dir,
                "elapsed_time": elapsed_time,
                "error_detail": error_detail
            }
    
    def _save_error_log(self, output_dir, error_message, error_detail, elapsed_time):
        """오류 로그 저장"""
        try:
            os.makedirs(output_dir, exist_ok=True)
            error_log_dir = os.path.join(output_dir, "error_logs")
            os.makedirs(error_log_dir, exist_ok=True)
            
            error_log_path = os.path.join(error_log_dir, "error.log")
            with open(error_log_path, "w", encoding='utf-8') as f:
                f.write(f"실험 실행 오류 로그\n")
                f.write(f"="*50 + "\n")
                f.write(f"오류 발생 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"실행 시간: {elapsed_time/60:.1f}분\n")
                f.write(f"오류 메시지: {error_message}\n\n")
                f.write(f"상세 오류 내용:\n")
                f.write(f"-"*50 + "\n")
                f.write(f"{error_detail}\n")
            
            print(f"📝 오류 로그 저장: {error_log_path}")
            
        except Exception as log_error:
            print(f"⚠️ 오류 로그 저장 실패: {str(log_error)}")
    
    def collect_metrics(self, experiment_results):
        """
        모든 실험 결과에서 메트릭 수집 및 통합
        
        Args:
            experiment_results (dict): 실험 결과 딕셔너리
            
        Returns:
            pandas.DataFrame: 통합된 메트릭 데이터프레임
        """
        print("📊 실험 메트릭 수집 중...")
        
        combined_metrics_df = pd.DataFrame()
        collected_count = 0
        
        for model_name, result in experiment_results.items():
            if result["status"] == "완료":
                metrics_file = os.path.join(result["output_dir"], "performance_metrics.csv")
                
                if os.path.exists(metrics_file):
                    try:
                        model_metrics = pd.read_csv(metrics_file)
                        
                        # 모델 이름이 없으면 추가
                        if 'Model' not in model_metrics.columns:
                            model_metrics['Model'] = model_name
                        
                        # 실험 정보 추가
                        model_metrics['Experiment_Status'] = 'Success'
                        model_metrics['Elapsed_Time'] = result.get("elapsed_time", 0)
                        
                        combined_metrics_df = pd.concat([combined_metrics_df, model_metrics], 
                                                       ignore_index=True)
                        collected_count += 1
                        
                        print(f"  ✅ {model_name}: {len(model_metrics)}개 사이클 메트릭 수집")
                        
                    except Exception as e:
                        print(f"  ❌ {model_name}: 메트릭 파일 읽기 오류 - {str(e)}")
                else:
                    print(f"  ⚠️ {model_name}: 메트릭 파일 없음")
            else:
                print(f"  💥 {model_name}: 실험 실패로 메트릭 없음")
        
        print(f"📈 총 {collected_count}개 모델의 메트릭 수집 완료")
        
        return combined_metrics_df
    
    def create_comparison_tables(self, combined_metrics_df, output_dir):
        """성능 비교 테이블 생성"""
        print("📊 성능 비교 테이블 생성 중...")
        
        if combined_metrics_df.empty:
            print("⚠️ 메트릭 데이터가 없어 비교 테이블을 생성할 수 없습니다")
            return
        
        try:
            # 메트릭별 비교 테이블 생성
            metrics_to_compare = ['mAP50', 'Precision', 'Recall', 'F1-Score', 
                                'Detected_Objects', 'Filtered_Objects']
            
            for metric in metrics_to_compare:
                if metric in combined_metrics_df.columns:
                    try:
                        # 사이클과 모델을 기준으로 피벗 테이블 생성
                        pivot_df = combined_metrics_df.pivot_table(
                            index='Cycle', 
                            columns='Model', 
                            values=metric,
                            aggfunc='mean'
                        )
                        
                        # 테이블 저장
                        table_file = os.path.join(output_dir, f"{metric}_comparison_table.csv")
                        pivot_df.to_csv(table_file)
                        
                        print(f"  📄 {metric} 비교 테이블 저장: {table_file}")
                        
                        # 최종 사이클 성능 요약
                        if not pivot_df.empty:
                            final_cycle = pivot_df.index.max()
                            final_performance = pivot_df.loc[final_cycle].sort_values(ascending=False)
                            
                            print(f"  🏆 {metric} 최종 성능 (사이클 {final_cycle}):")
                            for model, score in final_performance.head(3).items():
                                if not pd.isna(score):
                                    print(f"    {model}: {score:.4f}")
                        
                    except Exception as e:
                        print(f"  ❌ {metric} 테이블 생성 실패: {str(e)}")
                else:
                    print(f"  ⚠️ {metric} 컬럼이 없습니다")
            
            # 전체 메트릭 요약 테이블
            summary_file = os.path.join(output_dir, "performance_summary.csv")
            combined_metrics_df.to_csv(summary_file, index=False)
            print(f"📋 전체 성능 요약 저장: {summary_file}")
            
        except Exception as e:
            print(f"❌ 비교 테이블 생성 중 오류: {str(e)}")
    
    def generate_experiment_report(self, experiment_results, combined_metrics_df, output_dir):
        """실험 결과 종합 보고서 생성"""
        print("📝 실험 보고서 생성 중...")
        
        report_file = os.path.join(output_dir, "experiment_report.txt")
        
        try:
            with open(report_file, "w", encoding='utf-8') as f:
                # 보고서 헤더
                f.write("="*80 + "\n")
                f.write("ITERATIVE ACTIVE LEARNING 실험 보고서\n")
                f.write("="*80 + "\n")
                f.write(f"생성 일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"실험 설정:\n")
                f.write(f"  - 최대 사이클: {self.max_cycles}\n")
                f.write(f"  - 신뢰도 임계값: {self.conf_threshold}\n")
                f.write(f"  - IoU 임계값: {self.iou_threshold}\n")
                f.write(f"  - 분류 신뢰도 임계값: {self.class_conf_threshold}\n")
                f.write(f"  - GPU: {self.gpu_num}\n")
                f.write("\n")
                
                # 실험 결과 요약
                total_models = len(experiment_results)
                successful_count = sum(1 for r in experiment_results.values() if r["status"] == "완료")
                failed_count = total_models - successful_count
                
                f.write("📊 실험 결과 요약\n")
                f.write("-"*50 + "\n")
                f.write(f"총 모델 수: {total_models}개\n")
                f.write(f"성공: {successful_count}개 ({successful_count/total_models*100:.1f}%)\n")
                f.write(f"실패: {failed_count}개 ({failed_count/total_models*100:.1f}%)\n")
                f.write("\n")
                
                # 각 모델별 상세 결과
                f.write("📋 모델별 상세 결과\n")
                f.write("-"*50 + "\n")
                
                for model_name, result in experiment_results.items():
                    status = result["status"]
                    elapsed_time = result.get("elapsed_time", 0)
                    
                    f.write(f"\n🔹 {model_name}\n")
                    f.write(f"   상태: {status}\n")
                    f.write(f"   실행시간: {elapsed_time/60:.1f}분\n")
                    
                    if status == "완료":
                        f.write(f"   결과 디렉토리: {result['output_dir']}\n")
                        f.write(f"   완료 사이클: {result.get('cycles_completed', 'N/A')}\n")
                    else:
                        f.write(f"   오류 메시지: {result['message']}\n")
                
                # 성능 통계 (메트릭이 있는 경우)
                if not combined_metrics_df.empty:
                    f.write("\n\n📈 성능 통계\n")
                    f.write("-"*50 + "\n")
                    
                    # 최종 사이클의 평균 성능
                    final_cycle = combined_metrics_df['Cycle'].max()
                    final_metrics = combined_metrics_df[combined_metrics_df['Cycle'] == final_cycle]
                    
                    if not final_metrics.empty:
                        f.write(f"최종 사이클 ({final_cycle}) 평균 성능:\n")
                        
                        for metric in ['mAP50', 'Precision', 'Recall', 'F1-Score']:
                            if metric in final_metrics.columns:
                                avg_score = final_metrics[metric].mean()
                                f.write(f"  {metric}: {avg_score:.4f}\n")
                        
                        f.write(f"\n객체 탐지 통계:\n")
                        if 'Detected_Objects' in final_metrics.columns:
                            total_detected = final_metrics['Detected_Objects'].sum()
                            f.write(f"  총 탐지 객체: {total_detected:,}개\n")
                        
                        if 'Filtered_Objects' in final_metrics.columns:
                            total_filtered = final_metrics['Filtered_Objects'].sum()
                            f.write(f"  필터링된 객체: {total_filtered:,}개\n")
                            
                            if total_detected > 0:
                                filter_rate = total_filtered / (total_detected + total_filtered) * 100
                                f.write(f"  필터링 비율: {filter_rate:.1f}%\n")
                    
                    # 최고 성능 모델
                    if 'F1-Score' in combined_metrics_df.columns:
                        best_performance = combined_metrics_df.loc[combined_metrics_df['F1-Score'].idxmax()]
                        f.write(f"\n🏆 최고 성능 모델:\n")
                        f.write(f"  모델: {best_performance['Model']}\n")
                        f.write(f"  사이클: {best_performance['Cycle']}\n")
                        f.write(f"  F1-Score: {best_performance['F1-Score']:.4f}\n")
                
                # 실험 파일 목록
                f.write(f"\n\n📁 생성된 파일\n")
                f.write("-"*50 + "\n")
                f.write(f"- 실험 보고서: {report_file}\n")
                f.write(f"- 성능 요약: {os.path.join(output_dir, 'performance_summary.csv')}\n")
                f.write(f"- 비교 테이블: {output_dir}/*_comparison_table.csv\n")
                f.write(f"- 모델별 결과: {output_dir}/[model_name]/\n")
                
            print(f"📄 실험 보고서 저장: {report_file}")
            
        except Exception as e:
            print(f"❌ 보고서 생성 실패: {str(e)}")
    
    def save_experiment_config(self, output_dir):
        """실험 설정을 JSON 파일로 저장"""
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
            print(f"⚙️ 실험 설정 저장: {config_file}")
        except Exception as e:
            print(f"⚠️ 설정 저장 실패: {str(e)}")
    
    def run_iterative_experiments(self):
        """
        모든 YOLO 모델에 대해 Iterative 실험 실행
        
        Returns:
            dict: 종합 실험 결과
        """
        print("="*80)
        print("🚀 ITERATIVE ACTIVE LEARNING 실험 시작")
        print("="*80)
        
        # 실험 시작 시간 기록
        total_start_time = time.time()
        
        # 타임스탬프를 이용한 실험 디렉토리 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_dir = os.path.join(self.output_dir, f"iterative_experiment_{timestamp}")
        os.makedirs(experiment_dir, exist_ok=True)
        
        print(f"📁 실험 결과 저장 디렉토리: {experiment_dir}")
        
        # 실험 설정 저장
        self.save_experiment_config(experiment_dir)
        
        # YOLO 모델 목록 가져오기
        try:
            model_paths = self.get_yolo_models()
        except Exception as e:
            print(f"❌ YOLO 모델 로딩 실패: {str(e)}")
            return {"error": str(e)}
        
        # 실험 결과 저장용 딕셔너리
        experiment_results = {}
        
        # 진행률 표시를 위한 tqdm
        print(f"\n🔄 {len(model_paths)}개 모델에 대한 실험 시작")
        
        model_pbar = tqdm(model_paths, desc="Models", unit="model")
        
        # 각 YOLO 모델에 대해 실험 수행
        for i, model_path in enumerate(model_pbar):
            model_filename = os.path.basename(model_path)
            model_name = os.path.splitext(model_filename)[0]
            
            # 진행률 업데이트
            model_pbar.set_description(f"Processing {model_name}")
            
            print(f"\n🎯 실험 진행: {i+1}/{len(model_paths)} - {model_name}")
            
            # 개별 모델 실험 실행
            result = self.run_single_experiment(model_path)
            experiment_results[model_name] = result
            
            # 중간 결과 출력
            if result["status"] == "완료":
                print(f"✅ {model_name} 완료 ({result['elapsed_time']/60:.1f}분)")
            else:
                print(f"❌ {model_name} 실패: {result['message']}")
        
        model_pbar.close()
        
        # 전체 실험 시간 계산
        total_elapsed_time = time.time() - total_start_time
        
        print(f"\n⏰ 전체 실험 시간: {total_elapsed_time/60:.1f}분")
        
        # 메트릭 수집 및 통합
        print("\n📊 실험 결과 분석 중...")
        combined_metrics_df = self.collect_metrics(experiment_results)
        
        # 통합 메트릭 저장
        if not combined_metrics_df.empty:
            combined_metrics_file = os.path.join(experiment_dir, "combined_performance_metrics.csv")
            combined_metrics_df.to_csv(combined_metrics_file, index=False)
            print(f"💾 통합 메트릭 저장: {combined_metrics_file}")
            
            # 비교 테이블 생성
            self.create_comparison_tables(combined_metrics_df, experiment_dir)
        
        # 종합 보고서 생성
        self.generate_experiment_report(experiment_results, combined_metrics_df, experiment_dir)
        
        # 최종 결과 요약
        successful_count = sum(1 for r in experiment_results.values() if r["status"] == "완료")
        failed_count = len(experiment_results) - successful_count
        
        print("\n" + "="*80)
        print("🎉 ITERATIVE ACTIVE LEARNING 실험 완료")
        print("="*80)
        print(f"📊 실험 결과:")
        print(f"  - 총 모델: {len(model_paths)}개")
        print(f"  - 성공: {successful_count}개")
        print(f"  - 실패: {failed_count}개")
        print(f"  - 성공률: {successful_count/len(model_paths)*100:.1f}%")
        print(f"⏰ 총 실행 시간: {total_elapsed_time/60:.1f}분")
        print(f"📁 결과 저장 위치: {experiment_dir}")
        
        if not combined_metrics_df.empty:
            print(f"📈 메트릭 요약:")
            print(f"  - 통합 메트릭: {len(combined_metrics_df)}개 기록")
            print(f"  - 비교 테이블: {len([f for f in os.listdir(experiment_dir) if f.endswith('_comparison_table.csv')])}개")
        
        # 반환할 결과 딕셔너리 구성
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
        """임시 파일 정리"""
        print("🧹 임시 파일 정리 중...")
        
        temp_patterns = [
            "runs/detect/*/",  # YOLO 학습 임시 결과
            "*.yaml",          # 임시 YAML 파일
            "__pycache__/",    # Python 캐시
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
                    print(f"  🗑️ {len(temp_files)}개 {pattern} 파일/폴더 정리됨")
            except Exception as e:
                print(f"  ⚠️ {pattern} 정리 실패: {str(e)}")
        
        print("✅ 임시 파일 정리 완료")

def main():
    """테스트 실행을 위한 메인 함수"""
    # 테스트 설정
    processor = IterativeProcessor(
        yolo_models_dir="./models/initial_yolo",
        classifier_path="./models/classification/densenet121_100.pth",
        image_dir="./dataset/images",
        label_dir="./dataset/labels",
        output_dir="./results/iterative_process",
        conf_threshold=0.25,
        iou_threshold=0.5,
        class_conf_threshold=0.5,
        max_cycles=5,  # 테스트용으로 낮춤
        gpu_num=0
    )
    
    # 실험 실행
    results = processor.run_iterative_experiments()
    
    # 결과 출력
    if "error" not in results:
        print("\n🎯 실험 완료! 주요 결과:")
        summary = results["summary"]
        print(f"  - 성공률: {summary['success_rate']:.1f}%")
        print(f"  - 총 시간: {summary['total_time_minutes']:.1f}분")
        print(f"  - 시간당 실험: {summary['experiments_per_hour']:.1f}개")
    else:
        print(f"❌ 실험 실패: {results['error']}")

if __name__ == "__main__":
    main()