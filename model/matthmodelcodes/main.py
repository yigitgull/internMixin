import functions as f
import mathematical_model as m


time_limit = 30
counter_limit = 10
tolerance = 0.5
focus = 0
relaxed = False

m.main(time_limit=time_limit, counter_limit=counter_limit, tolerance=tolerance, focus=focus)
f.summarize_results("Outputs", "Output Summary.xlsx")
