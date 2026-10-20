##
## EPITECH PROJECT, 2026
## torch
## File description:
## Makefile
##

NAME1	=	my_torch_analyzer
NAME2	=	my_torch_generator

all:
	python3 -m zipapp src/analyzer -o $(NAME1) -p "/usr/bin/env python3"
	python3 -m zipapp src/generator -o $(NAME2) -p "/usr/bin/env python3"

re:	fclean all

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true

fclean:	clean
	$(RM) $(NAME)

.PHONY:	all re clean fclean
