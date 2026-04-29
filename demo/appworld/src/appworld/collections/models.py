import json
import os
from collections import defaultdict
from collections.abc import Sequence
from typing import Any, Literal, Self

from munch import Munch

from appworld.apps.lib.apis.local_remote import save_local_dbs, set_local_dbs
from appworld.apps.lib.models.db import CachedDBHandler, ModelHashHandler
from appworld.apps.lib.models.orm import SQLModel, record_field_difference
from appworld.common.collections import list_of, unique
from appworld.common.database import vacuum_sqlite_database
from appworld.common.errors import NoSuppressError
from appworld.common.imports import import_models_module
from appworld.common.io import safely_remove_file
from appworld.common.naming import modelize


def _validate_model_name(model_name: str) -> None:
    if model_name.count(".") != 1:
        raise Exception(f"Field name must be app_name.ClassName. Found {model_name}.")


class ModelCollection(Munch):
    @classmethod
    def load(
        cls,
        to_db_home_path: str | None = None,
        from_db_home_path: str | None = None,
        load_apps: Sequence[str] | None = None,
        force_empty_apps: Sequence[str] | None = None,
        skip_mandatory_apps: bool = False,
        create_db: bool = False,
    ) -> Self:
        if to_db_home_path is None and from_db_home_path is None:
            raise Exception("Either to_db_home_path or from_db_home_path must be given.")
        if to_db_home_path is None:
            to_db_home_path = from_db_home_path
        if from_db_home_path is None:
            from_db_home_path = to_db_home_path

        assert isinstance(from_db_home_path, str)  # mypy
        is_in_memory = "memory" in from_db_home_path
        if not is_in_memory and not os.path.exists(from_db_home_path):
            raise Exception(f"The from_db_home_path {from_db_home_path} does not exist.")
        if is_in_memory and force_empty_apps:
            raise Exception(
                "force_empty_apps must be None/empty when loading from an in-memory database."
            )
        if is_in_memory and not load_apps:
            raise Exception("load_apps must be given when loading from an in-memory database.")

        force_empty_apps = list(force_empty_apps or [])

        if load_apps is None:
            load_apps = [
                file_name.replace(".db", "")
                for file_name in os.listdir(from_db_home_path)
                if file_name.endswith(".db")
            ]
        load_apps = list(load_apps)
        load_apps = unique(load_apps)
        if not skip_mandatory_apps:
            mandatory_apps = ["admin", "supervisor"]
            for app in mandatory_apps:
                if app not in (load_apps + force_empty_apps):
                    load_apps.append(app)

        assert not (
            set(load_apps) & set(force_empty_apps)
        ), "force_empty_apps and load_apps must not intersect."

        for app_name in force_empty_apps:
            db_file_path = os.path.join(from_db_home_path, app_name + ".db")
            safely_remove_file(db_file_path)

        model_collection = cls()
        model_collection._from_db_home_path = from_db_home_path
        model_collection._to_db_home_path = to_db_home_path
        model_collection._ids_record_hashes = defaultdict(lambda: None)
        model_collection._ids = defaultdict(lambda: None)
        model_collection._record_hashes = defaultdict(lambda: None)
        for app_name in load_apps + force_empty_apps:
            models_module = import_models_module(app_name)
            model_collection[app_name] = models_module.SQLModel.name_to_model()
            model_collection[app_name].SQLModel = models_module.SQLModel
        model_collection.reset_db_home_path(create_db=create_db)
        ModelHashHandler.load(from_db_home_path, to_db_home_path)
        return model_collection

    # TODO: his function is a huge bottleneck. Remove it or make it more efficient. Not strickly needed really.
    # If I add it, note that each of the above setters in model_collection[app_name] call this function
    # which can make it extremely slow. At the least, I should use dict until the end above.
    # def __getitem__(self, key: str) -> dict:
    #     if key in self.app_names:
    #         self.reset_db_home_path(create_db=False)
    #     return super().__getitem__(key)

    @property
    def app_names(self) -> list[str]:
        return self.keys()

    def ids_record_hashes(self, app_name: str, model_name: str) -> list[tuple[int, str]]:
        ids_record_hashes_ = self._ids_record_hashes[(app_name, model_name)]
        if ids_record_hashes_ is not None:
            return ids_record_hashes_
        ids_record_hashes_ = self[app_name][model_name].find_all_take_fields(["id", "record_hash"])
        self._ids_record_hashes[(app_name, model_name)] = ids_record_hashes_
        return ids_record_hashes_

    def ids(self, app_name: str, model_name: str) -> set[int]:
        ids_ = self._ids[(app_name, model_name)]
        if ids_ is not None:
            return ids_
        ids_ = {
            id_record_hash[0] for id_record_hash in self.ids_record_hashes(app_name, model_name)
        }
        self._ids[(app_name, model_name)] = ids_
        return ids_

    def record_hashes(self, app_name: str, model_name: str) -> set[str]:
        record_hashes_ = self._record_hashes[(app_name, model_name)]
        if record_hashes_ is not None:
            return record_hashes_
        record_hashes_ = {
            id_record_hash[1] for id_record_hash in self.ids_record_hashes(app_name, model_name)
        }
        self._record_hashes[(app_name, model_name)] = record_hashes_
        return record_hashes_

    def reset_db_home_path(self, create_db: bool = False) -> None:
        set_local_dbs(
            self.to_db_home_path, self.from_db_home_path, self.app_names, create=create_db
        )

    def values(self) -> list[Any]:
        return [value for key, value in super().items() if not key.startswith("_")]

    def items(self) -> list[tuple[str, dict[str, Any]]]:
        return [(key, value) for key, value in super().items() if not key.startswith("_")]

    def keys(self) -> list[str]:
        return [key for key, value in super().items() if not key.startswith("_")]

    @property
    def from_db_home_path(self) -> str:
        return self._from_db_home_path

    @property
    def to_db_home_path(self) -> str:
        return self._to_db_home_path

    def vaccum(self) -> None:
        for app_name, models in self.items():
            db_home_path = models.SQLModel.db.home_path
            if "memory" in db_home_path:
                return
            db_path = os.path.join(db_home_path, app_name + ".db")
            vacuum_sqlite_database(db_path)

    def start_db_tracker(self) -> None:
        for models in self.values():
            models.SQLModel.start_db_tracker()

    def stop_db_tracker(self) -> None:
        for models in self.values():
            models.SQLModel.stop_db_tracker()

    def restart_db_tracker(self) -> None:
        for models in self.values():
            models.SQLModel.restart_db_tracker()

    def get_tracked_db_changes(self) -> dict[str, list[dict[str, Any]]]:
        tracked_db_changes: dict[str, list[dict[str, Any]]] = {}
        for app_name, models in self.items():
            tracked_db_changes[app_name] = models.SQLModel.get_tracked_db_changes()
        return tracked_db_changes

    def connections_are_open(self) -> bool:
        for models in self.values():
            if not models.SQLModel.db.connection_is_open():
                return False
        return True

    def save(
        self,
        db_home_path: str,
        format: Literal["full", "changes"] = "full",
        app_names: list[str] | None = None,
        delete_if_exists: bool = False,
        skip_mandatory_apps: bool = False,
        save_model_hashes: bool = False,
        vaccum: bool = False,
    ) -> None:
        app_names = app_names or self.app_names
        save_local_dbs(
            from_db_home_path=self.to_db_home_path,
            to_db_home_path=db_home_path,
            format=format,
            app_names=app_names,
            delete_if_exists=delete_if_exists,
            skip_mandatory_apps=skip_mandatory_apps,
            save_model_hashes=save_model_hashes,
            vaccum=vaccum,
        )

    def model_names(self) -> list[str]:
        return [name for name in self.keys() if name != "SQLModel"]

    def select(self, app_names: list[str] | None = None) -> Self:
        if app_names is not None:
            if set(self.model_names()) == set(app_names):
                return self
            for app_name in app_names:
                if app_name not in self:
                    raise Exception(f"App {app_name} is not in the model_collection.")
        else:
            app_names = list(self.model_names())
        model_collection = self.__class__(
            {
                app_name: name_to_class
                for app_name, name_to_class in self.items()
                if app_name in [*app_names, "SQLModel"]
            }
        )
        return model_collection

    def print_stats(self) -> None:
        print("App-wise Detailed Size Stats")
        full_stats = {
            app_name: {
                model_name: model_class.size()
                for model_name, model_class in models.SQLModel.name_to_model().items()
            }
            for app_name, models in self.items()
        }
        print(json.dumps(full_stats, indent=4))
        print("App-wise Aggregate Size Stats")
        aggregated_stats = {
            app_name: sum(stats_.values()) for app_name, stats_ in full_stats.items()
        }
        print(json.dumps(aggregated_stats, indent=4))
        empty_tables = [
            f"{app_name}.{table_name}"
            for app_name, tables_stats in full_stats.items()
            for table_name, size in tables_stats.items()
            if not size
        ]
        print("Empty Tables")
        print(json.dumps(empty_tables, indent=4))
        print(f"Global Aggregate Size: {sum(aggregated_stats.values())}")

    def model_hashes(self) -> dict[str, int | None]:
        model_hashes_: dict[str, int | None] = {}
        for app_name, name_to_model in self.items():
            name_to_model_hash = self[app_name].SQLModel.model_hashes()
            for model_name, _ in name_to_model.items():
                if model_name == "SQLModel":
                    continue
                name = f"{app_name}.{model_name}"
                hash_ = name_to_model_hash[model_name]
                model_hashes_[name] = hash_
        return model_hashes_

    def clear_model_hashes(self) -> None:
        ModelHashHandler.reset(self.to_db_home_path)

    def clear_ids_record_hashes(self) -> None:
        self._ids_record_hashes = defaultdict(lambda: None)
        self._ids = defaultdict(lambda: None)
        self._record_hashes = defaultdict(lambda: None)

    def close(self) -> None:
        CachedDBHandler.reset(self.to_db_home_path)
        self.clear_model_hashes()
        self.clear_ids_record_hashes()


class ModelCollectionPair:
    def __init__(
        self,
        start_db_home_path: str,
        start_model_collection: ModelCollection,
        end_db_home_path: str,
        end_model_collection: ModelCollection,
    ) -> None:
        # TODO: The ModelCollectionPair need not take both the db_home_path and model_collection.
        # only one of them should be enough. Refactor at some point.
        start_app_names = list(start_model_collection.keys())
        end_app_names = list(end_model_collection.keys())
        if set(start_app_names) != set(end_app_names):
            raise Exception(
                "The app names for model_collection_start for model_collection_end don't match."
            )
        self.app_names = start_app_names
        self.start_db_home_path = start_db_home_path
        self._start = start_model_collection
        self._end = end_model_collection
        self.end_db_home_path = end_db_home_path

    def merge(self, into: str) -> None:
        assert into in ["start", "end"], f"into must be either 'start' or 'end'. Found {into}."
        if into == "start":
            self._end = self._start
            self.end_db_home_path = self.start_db_home_path
        else:
            self._start = self._end
            self.start_db_home_path = self.end_db_home_path

    @classmethod
    def load(
        cls,
        start_db_home_path: str,
        end_db_home_path: str,
        load_apps: list[str] | None = None,
        force_empty_apps: list[str] | None = None,
    ) -> Self:
        start_model_collection = ModelCollection.load(
            to_db_home_path=start_db_home_path,
            from_db_home_path=start_db_home_path,
            load_apps=load_apps,
            force_empty_apps=force_empty_apps,
        )
        end_model_collection = ModelCollection.load(
            to_db_home_path=end_db_home_path,
            from_db_home_path=end_db_home_path,
            load_apps=load_apps,
            force_empty_apps=force_empty_apps,
        )
        return cls(
            start_db_home_path=start_db_home_path,
            start_model_collection=start_model_collection,
            end_db_home_path=end_db_home_path,
            end_model_collection=end_model_collection,
        )

    @property
    def start(self) -> ModelCollection:
        self._start.reset_db_home_path()
        return self._start

    @property
    def end(self) -> ModelCollection:
        self._end.reset_db_home_path()
        return self._end

    def no_suprresion_call(self, name: str, *args: Any, **kwargs: Any) -> Any:
        try:
            return getattr(self, name)(*args, **kwargs)
        except Exception as exception:
            message = (
                f"Error raised from {name}. If you encountered this while running "
                "the evaluation, this should not have happened. Please raise an issue on Github. "
                "The error message is:\n" + str(exception)
            )
            raise NoSuppressError(message) from exception

    def changed_model_names(
        self, include: list[str] | None = None, ignore: list[str] | None = None
    ) -> set[str]:
        return self.no_suprresion_call("_changed_model_names", include=include, ignore=ignore)

    def _changed_model_names(
        self, include: list[str] | None = None, ignore: list[str] | None = None
    ) -> set[str]:
        if isinstance(include, str):
            raise Exception(f"'include' must be a list of strings. Found {include}")

        if isinstance(ignore, str):
            raise Exception(f"'ignore' must be a list of strings. Found {ignore}")

        include_model_names = include or []
        ignore_model_names = ignore or []

        if "supervisor.Task" not in ignore_model_names:
            ignore_model_names.append("supervisor.Task")

        if "admin.PaymentCard" not in ignore_model_names:
            ignore_model_names.append("admin.PaymentCard")

        if "amazon.BrowsedProduct" not in ignore_model_names:
            # This was added after V1 and no task requires updating it, so ignoring it globally for now.
            ignore_model_names.append("amazon.BrowsedProduct")

        if include_model_names is None and ignore_model_names is None:
            raise Exception(
                "At least one of include_model_names and ignore_model_names must be not None."
            )

        start_app_names = list(self.start.keys())
        end_app_names = list(self.end.keys())
        if set(start_app_names) != set(end_app_names):
            raise Exception(
                "The app names for model_collection_start for model_collection_end don't match."
            )

        all_model_names: list[str] = []
        for app_name in start_app_names:
            app_start_model_names = self.start[app_name].SQLModel.model_names()
            app_end_model_names = self.end[app_name].SQLModel.model_names()
            assert (
                app_start_model_names == app_end_model_names
            ), f"The class names for {app_name} for model_collection_start for model_collection_end don't match."
            all_model_names += [f"{app_name}.{model_name}" for model_name in app_start_model_names]

        candidate_model_names = all_model_names

        if include_model_names:
            for model_name in include_model_names:
                assert (
                    model_name.count(".") == 1
                ), f"Field name must be app_name.ClassName. Found {model_name}."
            assert (model_name in all_model_names for model_name in include_model_names)
            candidate_model_names = include_model_names

        elif ignore_model_names:
            for model_name in ignore_model_names:
                assert (
                    model_name.count(".") == 1
                ), f"Field name must be app_name.ClassName. Found {model_name}."
            candidate_model_names = [
                model_name for model_name in all_model_names if model_name not in ignore_model_names
            ]

        candidate_model_names = [
            model_name
            for model_name in candidate_model_names
            if not model_name.endswith("ModelHash")
        ]

        start_model_hashes = self.start.model_hashes()
        end_model_hashes = self.end.model_hashes()
        for candidate_model_name in list(candidate_model_names):  # list(..) is to make a copy
            if start_model_hashes[candidate_model_name] == end_model_hashes[candidate_model_name]:
                candidate_model_names.remove(candidate_model_name)

        changed_model_names_: set[str] = set()
        for model_name in candidate_model_names:
            app_name, class_name = model_name.split(".")
            size_1 = getattr(self.start[app_name], class_name).size()
            size_2 = getattr(self.end[app_name], class_name).size()
            if size_1 != size_2:
                changed_model_names_.add(model_name)
                continue
            start_record_hashes = self.start.record_hashes(app_name, class_name)
            end_record_hashes = self.end.record_hashes(app_name, class_name)
            if set(start_record_hashes) != set(end_record_hashes):
                changed_model_names_.add(model_name)

        formatted_changed_model_names: set[str] = set()
        for model_name in changed_model_names_:
            app_name, table_name = model_name.split(".")
            class_name = modelize(table_name)
            formatted_changed_model_names.add(f"{app_name}.{class_name}")

        return formatted_changed_model_names

    def changed_records(
        self, model_name: str, updated_state: str = "end"
    ) -> tuple[list[SQLModel], list[SQLModel], list[SQLModel]]:
        return self.no_suprresion_call(
            "_changed_records", model_name=model_name, updated_state=updated_state
        )

    def _changed_records(
        self, model_name: str, updated_state: str = "end"
    ) -> tuple[list[SQLModel], list[SQLModel], list[SQLModel]]:
        """
        Returns list in the following order:
        added_records: records from end state that are not in start state (match by id)
        updated_records:
            records for which ID exists in both start and end state but some of the
            fields have been updated. This is computed by a hash-based difference
            between records. You can control whether you want to obtain these updated
            records in end or start state via 'updated_state' argument.
        removed_records: records from start state that are not in end state (match by id)
        EDIT: If a record exists in both added and deleted (matched by hash), they will cancel
        out each other and be removed from both added and deleted.
        """
        assert updated_state in [
            "start",
            "end",
        ], f"updated_state must be either 'start' or 'end'. Found {updated_state}."
        _validate_model_name(model_name)
        app_name, model_class_name = model_name.split(".")
        start_ids = self.start.ids(app_name, model_class_name)
        end_ids = self.end.ids(app_name, model_class_name)
        added_records = self.end[app_name][model_class_name].by_ids(end_ids - start_ids)
        removed_records = self.start[app_name][model_class_name].by_ids(start_ids - end_ids)
        if updated_state == "end":
            ids_hashes_1 = self.end.ids_record_hashes(app_name, model_class_name)
            ids_2 = self.start.ids(app_name, model_class_name)
            record_hashes_2 = self.start.record_hashes(app_name, model_class_name)
            klass = self.end[app_name][model_class_name]
        else:
            ids_hashes_1 = self.start.ids_record_hashes(app_name, model_class_name)
            ids_2 = self.end.ids(app_name, model_class_name)
            record_hashes_2 = self.end.record_hashes(app_name, model_class_name)
            klass = self.start[app_name][model_class_name]
        updated_ids = [
            id_hash[0]
            for id_hash in ids_hashes_1
            if (id_hash[1] not in record_hashes_2) and (id_hash[0] in ids_2)
        ]
        updated_records = klass.by_ids(updated_ids)
        # cancel out added and removed records that have identical .record_hash,
        # one entry should cancel only one other entry.
        record_hash_to_added_records = defaultdict(list)
        for record in added_records:
            record_hash_to_added_records[record.record_hash].append(record)
        record_hash_to_removed_records = defaultdict(list)
        for record in removed_records:
            record_hash_to_removed_records[record.record_hash].append(record)
        added_records = []
        removed_records = []
        added_or_removed_record_hashes = set(
            record_hash_to_added_records.keys() | record_hash_to_removed_records.keys()
        )
        for record_hash in added_or_removed_record_hashes:
            added_records_ = record_hash_to_added_records[record_hash]
            removed_records_ = record_hash_to_removed_records[record_hash]
            min_len = min(len(added_records_), len(removed_records_))
            added_records += added_records_[min_len:]
            removed_records += removed_records_[min_len:]
        return added_records, updated_records, removed_records

    def changed_field_names(
        self,
        model_name: str,
        id: int,
        ignore: list[str] | None = None,
    ) -> set[str] | None:
        return self.no_suprresion_call(
            "_changed_field_names", model_name=model_name, id=id, ignore=ignore
        )

    def _changed_field_names(
        self,
        model_name: str,
        id: int,
        ignore: list[str] | None = None,
    ) -> set[str] | None:
        _validate_model_name(model_name)
        app_name, model_class_name = model_name.split(".")
        start_record = self.start[app_name][model_class_name].by_id(id)
        end_record = self.end[app_name][model_class_name].by_id(id)
        if start_record is None or end_record is None:
            return None
        ignore = ignore or []
        changed_field_names_ = record_field_difference(end_record, start_record)
        changed_field_names_ = {
            field_name for field_name in changed_field_names_ if field_name not in ignore
        }
        return changed_field_names_

    def show_changes(self, model_name: str) -> None:
        _validate_model_name(model_name)
        added_records, updated_records, removed_records = self.changed_records(
            model_name=model_name
        )
        added_record_ids = list_of(added_records, "id")
        removed_record_ids = list_of(removed_records, "id")
        updated_record_ids = list_of(updated_records, "id")
        print(f"In {model_name}")
        print(f"Added record ids: {added_record_ids}")
        print(f"Removed record ids: {removed_record_ids}")
        print(f"Updated record ids: {updated_record_ids}")
        for updated_record_id in updated_record_ids:
            updated_fields = self.changed_field_names(model_name=model_name, id=updated_record_id)
            print(f"Updated fields for record id {updated_record_id}: {updated_fields}")
