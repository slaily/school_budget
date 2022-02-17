runserver:
	python school_budget_web/app.py
dump_dependencies:
	pipenv run pip freeze > requirements.txt
run_notebooks:
	jupyter notebook
