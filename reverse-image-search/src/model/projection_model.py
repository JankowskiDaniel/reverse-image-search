from torch import nn

from ..settings import PROJECTION_DIM

class ProjectionHead(nn.Module):
    def __init__(
        self, embedding_dim: int, project_dim: int = PROJECTION_DIM, dropout: float = 0.2
    ) -> None:
        super(ProjectionHead, self).__init__()
        self.l1 = nn.Linear(embedding_dim, project_dim)
        self.gelu = nn.GELU()
        self.l2 = nn.Linear(project_dim, project_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(project_dim)

    def forward(self, embedding):
        output = self.l1(embedding)
        embedding = self.gelu(output)
        embedding = self.l2(embedding)
        embedding = self.dropout(embedding)
        embedding = embedding + output
        embedding = self.norm(embedding)
        return embedding