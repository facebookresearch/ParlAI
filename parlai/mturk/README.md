`parlai/mturk` contains a few separate MTurk-related folders:

- **core**: contains all the files that are used to make the underlying MTurk interaction work properly
- **tasks**: contains a number of tasks that have been launched on MTurk to collect various datasets (shared for reproducibility), as well as a few tasks that exist as examples of possible MTurk functionality
- **scripts/**: contains various helpful review and cleanup scripts to manage tasks from the command line (especially after mistakes or major failures).
- **webapp**: contains server and build files for the [Beta] ParlAI-MTurk frontend, which is currently under heavy development and may undergo many changes
- **run_data/**: contains all of the cross-run state in sqlite databases (default `pmt_data.db` and `pmt_sbdata.db`), as well as any world data saved by the default behavior of `MTurkDataWorld` (in `live` and `sandbox` folders, following the behavior specified in `MTurkDataHandler`).
