##
## EPITECH PROJECT, 2026
## torch
## File description:
## Makefile
##

NAME1	=	my_torch_analyzer
NAME2	=	my_torch_generator

all:
	python3 -m zipapp src -m analyzer:main -o $(NAME1) -p "/usr/bin/env python3"
	python3 -m zipapp src -m generator:main -o $(NAME2) -p "/usr/bin/env python3"

re:	fclean all

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true

fclean:	clean
	$(RM) $(NAME1) $(NAME2)

.PHONY:	all re clean fclean
