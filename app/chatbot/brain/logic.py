import csv
import logging
import os
import re
from difflib import get_close_matches
from typing import Optional, Tuple

import nltk
from nltk.inference import ResolutionProver, TableauProver
from nltk.sem import Expression

nltk.download("punkt", quiet=True)

logger = logging.getLogger(__name__)

read_expr = Expression.fromstring


class KnowledgeBase:
    def __init__(self, kb_path: Optional[str] = None):
        self.kb_statements = []
        self.kb_expressions = []
        self.matching_entities = {}

        if kb_path is None:
            kb_path = os.path.join(
                os.path.dirname(__file__), "..", "knowledge", "knowledge_base.csv"
            )

        self.kb_path = kb_path
        self.load_kb()
        self.check_consistency()

    def load_kb(self) -> None:
        self.kb_statements = []

        try:
            with open(self.kb_path, "r", encoding="utf-8") as file:
                reader = csv.DictReader(file)
                for row in reader:
                    statement = row.get("statement", "").strip('"')
                    if statement:
                        self.kb_statements.append(statement)
                        self.update_matching_entities_from_statement(statement)

            self.kb_expressions = [read_expr(stmt) for stmt in self.kb_statements]
            logger.info(
                f"Loaded {len(self.kb_expressions)} statements from knowledge base"
            )

        except Exception as e:
            logger.error(f"Error loading knowledge base: {e}")
            self.kb_statements = []
            self.kb_expressions = []

    def update_matching_entities_from_statement(self, statement: str) -> None:
        m = re.search(r"\((.*?)\)", statement)
        if m:
            args_str = m.group(1)
            args = [arg.strip() for arg in args_str.split(",")]
            for arg in args:
                candidate = "".join(arg.split())
                candidate_key = candidate.lower()
                if candidate_key not in self.matching_entities:
                    self.matching_entities[candidate_key] = candidate

    def format_entity(self, entity_str: str) -> str:
        candidate = "".join(entity_str.strip().split())
        candidate_key = candidate.lower()
        possibilities = list(self.matching_entities.keys())
        matches = get_close_matches(candidate_key, possibilities, n=1, cutoff=0.8)
        if matches:
            return self.matching_entities[matches[0]]
        else:
            self.matching_entities[candidate_key] = candidate
            return candidate

    def check_consistency(self) -> bool:
        prover = TableauProver()
        for i, expr1 in enumerate(self.kb_expressions):
            for expr2 in self.kb_expressions[i + 1 :]:
                if (
                    hasattr(expr1, "pred")
                    and hasattr(expr2, "pred")
                    and expr1.pred == expr2.pred  # type: ignore
                ):
                    try:
                        negated_expr2 = read_expr(f"~({expr2})")
                        if prover.prove(expr1, [negated_expr2]):
                            logger.warning(f"Contradiction found: {expr1} and {expr2}")
                            return False
                    except Exception as e:
                        logger.error(f"Error checking contradiction: {e}")
        logger.info("Knowledge base is consistent")
        return True

    def check_consistency_with_new_statement(self, statement: str) -> bool:
        try:
            expr = read_expr(statement)
            temp_kb = self.kb_expressions.copy()
            temp_kb.append(expr)
            prover = TableauProver()
            for i, expr1 in enumerate(temp_kb):
                for expr2 in temp_kb[i + 1 :]:
                    try:
                        negated_expr2 = read_expr(f"~({expr2})")
                        if prover.prove(expr1, [negated_expr2]):
                            return False
                    except Exception as e:
                        logger.error(f"Error in consistency check: {e}")
            return True
        except Exception as e:
            logger.error(f"Error checking consistency: {e}")
            return False

    def add_statement(self, statement: str) -> Tuple[bool, str]:
        try:
            if statement in self.kb_statements:
                return True, "This statement is already in my knowledge base."
            if not self.check_consistency_with_new_statement(statement):
                return False, "This statement contradicts my existing knowledge."
            new_expr = read_expr(statement)
            self.kb_statements.append(statement)
            self.kb_expressions.append(new_expr)
            # Update matching entities with the new statement's entities.
            self.update_matching_entities_from_statement(statement)
            return True, "OK, I will remember that"
        except Exception as e:
            logger.error(f"Error adding statement: {e}")
            return False, f"I couldn't understand that statement. Error: {e}"

    def save_kb(self) -> bool:
        try:
            with open(self.kb_path, "w", encoding="utf-8", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["statement"])
                for statement in self.kb_statements:
                    writer.writerow([f'"{statement}"'])
            return True
        except Exception as e:
            logger.error(f"Error saving knowledge base: {e}")
            return False

    def debug_knowledge_base(self) -> str:
        output = ["Knowledge Base Contents:"]
        for i, statement in enumerate(self.kb_statements):
            output.append(f"{i + 1}: {statement}")
        return "\n".join(output)

    def check_statement(self, statement: str) -> Tuple[str, str]:
        try:
            expr = read_expr(statement)
            negated_expr = read_expr(f"~({statement})")
            prover = ResolutionProver()
            if prover.prove(expr, self.kb_expressions):
                return "Correct", "I can prove this is true based on my knowledge."
            if prover.prove(negated_expr, self.kb_expressions):
                return "Incorrect", "I can prove this is false based on my knowledge."
            return (
                "I don't know",
                "I don't have enough information to determine if this is true or false.",
            )
        except Exception as e:
            logger.error(f"Error checking statement: {e}")
            return "I don't know", f"I couldn't understand that statement. Error: {e}"

    def parse_natural_language(self, text: str) -> Optional[str]:
        text = text.strip()
        logger.info(f"Parsing natural language: '{text}'")

        # Use self.format_entity to normalize entity names using fuzzy matching.
        # The regex patterns use re.IGNORECASE to match regardless of case.

        if re.match(r"(.*) is an album by (.*)", text, re.IGNORECASE):
            parts = re.match(r"(.*) is an album by (.*)", text, re.IGNORECASE)
            album = self.format_entity(parts.group(1))  # type: ignore
            artist = self.format_entity(parts.group(2))  # type: ignore
            result = f"AlbumBy({album}, {artist})"
            return result

        if re.match(r"(.*) is a track on (.*)", text, re.IGNORECASE):
            parts = re.match(r"(.*) is a track on (.*)", text, re.IGNORECASE)
            track = self.format_entity(parts.group(1))  # type: ignore
            album = self.format_entity(parts.group(2))  # type: ignore
            result = f"TrackOn({track}, {album})"
            return result

        if re.match(r"(.*) collaborates with (.*)", text, re.IGNORECASE):
            parts = re.match(r"(.*) collaborates with (.*)", text, re.IGNORECASE)
            artist1 = self.format_entity(parts.group(1))  # type: ignore
            artist2 = self.format_entity(parts.group(2))  # type: ignore
            return f"Collaborates({artist1}, {artist2})"

        if re.match(r"(.*) is signed to (.*)", text, re.IGNORECASE):
            parts = re.match(r"(.*) is signed to (.*)", text, re.IGNORECASE)
            artist = self.format_entity(parts.group(1))  # type: ignore
            label = self.format_entity(parts.group(2))  # type: ignore
            return f"SignedTo({artist}, {label})"

        if re.match(r"(.*) creates (.*)", text, re.IGNORECASE):
            parts = re.match(r"(.*) creates (.*)", text, re.IGNORECASE)
            artist = self.format_entity(parts.group(1))  # type: ignore
            creation = self.format_entity(parts.group(2))  # type: ignore
            return f"Creates({artist}, {creation})"

        if re.match(r"(.*) is the creator of (.*)", text, re.IGNORECASE):
            parts = re.match(r"(.*) is the creator of (.*)", text, re.IGNORECASE)
            artist = self.format_entity(parts.group(1))  # type: ignore
            creation = self.format_entity(parts.group(2))  # type: ignore
            return f"Creates({artist}, {creation})"

        if re.match(r"(.*) has fans", text, re.IGNORECASE):
            parts = re.match(r"(.*) has fans", text, re.IGNORECASE)
            artist = self.format_entity(parts.group(1))  # type: ignore
            return f"HasFans({artist})"

        if re.match(r"(.*) is a rap artist", text, re.IGNORECASE):
            parts = re.match(r"(.*) is a rap artist", text, re.IGNORECASE)
            artist = self.format_entity(parts.group(1))  # type: ignore
            return f"RapArtist({artist})"

        if re.match(r"(.*) is a pop artist", text, re.IGNORECASE):
            parts = re.match(r"(.*) is a pop artist", text, re.IGNORECASE)
            artist = self.format_entity(parts.group(1))  # type: ignore
            return f"PopArtist({artist})"

        if re.match(r"(.*) is a music artist", text, re.IGNORECASE):
            parts = re.match(r"(.*) is a music artist", text, re.IGNORECASE)
            artist = self.format_entity(parts.group(1))  # type: ignore
            return f"MusicArtist({artist})"

        if re.match(r"(.*) is a popular artist", text, re.IGNORECASE):
            parts = re.match(r"(.*) is a popular artist", text, re.IGNORECASE)
            artist = self.format_entity(parts.group(1))  # type: ignore
            return f"PopularArtist({artist})"

        if re.match(r"(.*) is a track", text, re.IGNORECASE):
            parts = re.match(r"(.*) is a track", text, re.IGNORECASE)
            track = self.format_entity(parts.group(1))  # type: ignore
            return f"Track({track})"

        if re.match(r"(.*) is an album", text, re.IGNORECASE):
            parts = re.match(r"(.*) is an album", text, re.IGNORECASE)
            album = self.format_entity(parts.group(1))  # type: ignore
            return f"Album({album})"

        if re.match(r"(.*) is an? (.*)", text, re.IGNORECASE):
            parts = re.match(r"(.*) is an? (.*)", text, re.IGNORECASE)
            subject = self.format_entity(parts.group(1))  # type: ignore
            predicate = self.format_entity(parts.group(2))  # type: ignore
            return f"{predicate}({subject})"

        if re.match(r"(.*) is not an? (.*)", text, re.IGNORECASE):
            parts = re.match(r"(.*) is not an? (.*)", text, re.IGNORECASE)
            subject = self.format_entity(parts.group(1))  # type: ignore
            predicate = self.format_entity(parts.group(2))  # type: ignore
            return f"~{predicate}({subject})"

        if re.match(r"(.*) does not collaborate with (.*)", text, re.IGNORECASE):
            parts = re.match(
                r"(.*) does not collaborate with (.*)", text, re.IGNORECASE
            )
            artist1 = self.format_entity(parts.group(1))  # type: ignore
            artist2 = self.format_entity(parts.group(2))  # type: ignore
            return f"~Collaborates({artist1}, {artist2})"

        if re.match(r"(.*) is not signed to (.*)", text, re.IGNORECASE):
            parts = re.match(r"(.*) is not signed to (.*)", text, re.IGNORECASE)
            artist = self.format_entity(parts.group(1))  # type: ignore
            label = self.format_entity(parts.group(2))  # type: ignore
            return f"~SignedTo({artist}, {label})"

        if re.match(r"(.*) does not create (.*)", text, re.IGNORECASE):
            parts = re.match(r"(.*) does not create (.*)", text, re.IGNORECASE)
            artist = self.format_entity(parts.group(1))  # type: ignore
            creation = self.format_entity(parts.group(2))  # type: ignore
            return f"~Creates({artist}, {creation})"

        logger.warning(f"No pattern matched: '{text}'")
        return None

    def handle_logic_command(self, command: str) -> str:
        command = command.strip()
        logger.info(f"Processing logic command: {command}")

        def get_example_formats():
            return (
                "Try using one of these formats:\n"
                "- X is a RapArtist (e.g., 'Drake is a RapArtist')\n"
                "- X is a Track (e.g., 'Sicko Mode is a Track')\n"
                "- X is an album by Y (e.g., 'Take Care is an album by Drake')\n"
                "- X is a track on Y (e.g., 'Sicko Mode is a track on Astroworld')\n"
                "- X collaborates with Y (e.g., 'Drake collaborates with Travis Scott')\n"
                "- X is signed to Y (e.g., 'Drake is signed to OVO')\n"
                "- X creates Y (e.g., 'Drake creates Take Care')\n"
                "- X has fans (e.g., 'Drake has fans')"
            )

        if command.lower().startswith("i know that"):
            statement = command[12:].strip()
            logger.info(f"Extracted statement: '{statement}'")
            fol_statement = self.parse_natural_language(statement)
            if not fol_statement:
                return f"I couldn't understand that statement. {get_example_formats()}"
            logger.info(f"Parsed FOL: {fol_statement}")
            success, message = self.add_statement(fol_statement)
            if success:
                return f"{message}. {statement}."
            else:
                return message

        elif command.lower().startswith("check that"):
            statement = command[11:].strip()
            logger.info(f"Extracted statement: '{statement}'")
            fol_statement = self.parse_natural_language(statement)
            if not fol_statement:
                return f"I couldn't understand that statement. {get_example_formats()}"
            logger.info(f"Parsed FOL: {fol_statement}")
            result, explanation = self.check_statement(fol_statement)
            return f"{result}. {explanation}"

        elif command.lower() == "save knowledge base":
            if self.save_kb():
                return "Knowledge base saved successfully."
            else:
                return "Error saving knowledge base."

        elif command.lower() == "debug knowledge base":
            return self.debug_knowledge_base()

        elif command.lower() == "help with knowledge base":
            return (
                "You can interact with my music knowledge base using these commands:\n\n"
                "1. To check if something is true: 'Check that X is Y'\n"
                "2. To teach me something new: 'I know that X is Y'\n"
                "3. To save what you've taught me: 'Save knowledge base'\n\n"
                f"{get_example_formats()}"
            )

        else:
            return (
                "I don't understand that command. Please use:\n"
                "- 'I know that X is Y' to teach me something\n"
                "- 'Check that X is Y' to check a fact\n"
                "- 'Save knowledge base' to save what you've taught me\n"
                "- 'Help with knowledge base' for examples"
            )
