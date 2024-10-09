from flask import Blueprint, request, jsonify
from flask_restx import Api, Resource, fields
from ..models.neat_model import NEATModel
from .. import db
import json

api_bp = Blueprint("api", __name__)
api = Api(
    api_bp, version="1.0", title="NEAT Model API", description="A simple NEAT Model API"
)

model_fields = api.model(
    "NEATModel",
    {
        "model_name": fields.String(required=True, description="The name of the model"),
        "dataset": fields.String(required=True, description="The dataset used"),
        "version": fields.String(required=True, description="The version of the model"),
        "raw_data": fields.String(
            required=True, description="The raw data of the model"
        ),
        "parsed_model": fields.Raw(required=True, description="The parsed model data"),
    },
)


@api.route("/models")
class ModelList(Resource):
    @api.doc("list_models")
    def get(self):
        models = NEATModel.query.all()
        ret = [model.to_dict() for model in models]

        return ret

    @api.doc("create_model")
    @api.expect(model_fields)
    @api.marshal_with(model_fields, code=201)
    def post(self):
        data = api.payload
        parsed_model = json.loads(data["parsed_model"])
        print(parsed_model)
        new_model = NEATModel(
            model_name=data["model_name"],
            dataset=data["dataset"],
            version=data["version"],
            raw_data=data["raw_data"].encode(),
            parsed_model=parsed_model,
        )
        db.session.add(new_model)
        db.session.commit()
        return new_model.to_dict(), 201


@api.route("/models/<int:id>")
@api.param("id", "The model identifier")
class Model(Resource):
    @api.doc("get_model")
    @api.marshal_with(model_fields)
    def get(self, id):
        return NEATModel.query.get_or_404(id).to_dict()

    @api.doc("update_model")
    @api.expect(model_fields)
    @api.marshal_with(model_fields)
    def put(self, id):
        model = NEATModel.query.get_or_404(id)
        data = api.payload
        parsed_model = json.loads(data["parsed_model"])
        print(parsed_model)
        model.model_name = data.get("model_name", model.model_name)
        model.dataset = data.get("dataset", model.dataset)
        model.version = data.get("version", model.version)
        if "raw_data" in data:
            model.raw_data = data["raw_data"].encode()
        if "parsed_model" in data:
            model.parsed_model = parsed_model
        db.session.commit()
        return model.to_dict()

    @api.doc("delete_model")
    @api.response(204, "Model deleted")
    def delete(self, id):
        model = NEATModel.query.get_or_404(id)
        db.session.delete(model)
        db.session.commit()
        return "", 204
