from ._state import (
    DataframeStore,
    bind_store,
    get_active_df,
    set_active_df,
    get_active_name,
    check_columns,
    _active_store_ctx,
)
from .data_tools import (
    load_dataset,
    describe_dataset,
    filter_data,
    aggregate_data,
    correlation_analysis,
)
from .fetch_tools import (
    load_hf_dataset,
    list_available_datasets,
    fetch_kaggle_dataset,
    search_kaggle_datasets,
)
from .viz_tools import create_visualization, PLOTS_DIR
from .report_tools import generate_report

ALL_TOOLS = [
    load_hf_dataset,
    list_available_datasets,
    fetch_kaggle_dataset,
    search_kaggle_datasets,
    load_dataset,
    describe_dataset,
    filter_data,
    aggregate_data,
    correlation_analysis,
    create_visualization,
    generate_report,
]
