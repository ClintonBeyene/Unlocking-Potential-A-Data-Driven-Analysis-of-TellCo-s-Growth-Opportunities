# Unlocking-Potential-A-Data-Driven-Analysis-of-TellCo-s-Growth-Opportunities
## Overview
This project aims to analyze TellCo, a mobile service provider, to identify growth and profitability opportunities. By exploring a telecom dataset, we extract insights into user behavior, engagement, experience, and satisfaction. The goal is to provide recommendations on whether TellCo is worth buying or selling.

## Project Structure
- **src/**: Main Python scripts for data analysis and dashboard creation.
- **notebooks/**: Jupyter notebooks for exploratory data analysis and documentation.
- **tests/**: Unit tests for code reliability.
- **scripts/**: Additional scripts for data preprocessing.
- **requirements.txt**: Python dependencies.
- **README.md**: Project objectives, tasks, and instructions.
- **.gitignore**: Files and directories to ignore in version control.
- **.github/workflows/unittests.yml**: GitHub Actions for unit tests.
- **Dockerfile**: Configuration for building the project as a Docker image.
## Project Tasks and Methodology
### User Overview Analysis:
- Identify top handsets and manufacturers.
- Analyze user behavior on various applications.
### User Engagement Analysis:
- Track engagement metrics (session frequency, duration, total traffic).
- Classify customers into engagement groups using k-means clustering.
### Experience Analytics:
- Aggregate network parameters and device characteristics.
- Analyze TCP retransmission, RTT, throughput, and handset type.
- Segment users based on experience metrics using k-means clustering.
### Satisfaction Analysis:
- Assign engagement and experience scores.
- Calculate satisfaction scores and predict them using regression modeling.
- Perform k-means clustering on engagement and experience scores.
- Export user data with scores to a local MySQL database.
## Running the Project
1. Clone the repository.
2. Install dependencies: pip install -r requirements.txt.
3. Run unit tests.
4. Execute main Python scripts for data analysis and dashboard creation.
5. Deploy the Docker image.
## Conclusion
The Telecom Profit Analysis project leverages data-driven insights to inform investment decisions regarding TellCo. By analyzing user behavior, engagement, experience, and satisfaction, the project uncovers opportunities for growth and profitability in the telecommunications industry.

## Author: Clinton Beyene