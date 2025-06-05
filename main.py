import monitor
import analysis


def main():
    monitor.main()
    analyzer = analysis.SortingAlgorithmAnalyzer()
    if analyzer.load_data():
        analyzer.analyze_theory_vs_practice_gap()
        analyzer.analyze_cache_effects()
        analyzer.analyze_practical_performance_zones()
        analyzer.analyze_compiler_optimization_impact()
        analyzer.generate_research_oriented_report()
    else:
        print("Data loading failed. Please check the data files.")


if __name__ == "__main__":
    main()
