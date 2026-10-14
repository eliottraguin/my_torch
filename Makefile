##
## EPITECH PROJECT, 2026
## torch
## File description:
## Makefile
##

NAME	=	my_torch_analyzer

all:
	python3 -m zipapp src/ -o $(NAME) -p "/usr/bin/env python3"

re:	fclean all

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true

fclean:	clean
	$(RM) $(NAME)

.PHONY:	all re clean fclean
