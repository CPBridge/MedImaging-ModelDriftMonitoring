# Drift results with final methods


## Naming Scheme:
### drift_analysis_allpoc_emd_jackknife_helllinger_final
- Run on all points of care
- Uses the Earth Mover's Distance for all continuous variables, the categorical variables use Hellinger Distance
- Uses Jackknife Resampling

### drift_analysis_allpoc_emd_jackknife_helllinger_final_badq05
- same as above 
- additionally here the reference window is oversampled with images from the worst 0.05% of images
    --good_q 0.05 \
    --good_start_date "2019-10-01" \
    --good_end_date "2020-01-01" \
- Note: The samples here are drawn from the whole 2 year period and injected only into the reference window, should this be repeated where we also only draw from the reference window?


## Analysis code:
The code that I used for all the plotting is available here: 
- The plotting script is at: src/scripts/analysis/basic_performance_plots.py