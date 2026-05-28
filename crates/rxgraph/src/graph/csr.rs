use anyhow::Result;

use super::repo::{EdgeId, NodeId};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Csr {
    pub(crate) offsets: Vec<usize>,
    pub(crate) edge_ids: Vec<EdgeId>,
    pub(crate) dests: Vec<NodeId>,
}

/// Constructs a CSR (Compressed Sparse Row) data structure for outgoing edges.
pub(crate) fn build_csr(node_count: usize, edges: &[(NodeId, NodeId)]) -> Result<Csr> {
    let mut offsets = vec![0usize; node_count + 1];

    for &(src, _dest) in edges {
        offsets[src as usize + 1] += 1;
    }

    for i in 1..offsets.len() {
        offsets[i] += offsets[i - 1];
    }

    let mut edge_ids = vec![0; edges.len()];
    let mut dests = vec![0; edges.len()];
    let mut cursor = offsets.clone();

    for (edge_id, &(src, dest)) in edges.iter().enumerate() {
        let pos = cursor[src as usize];
        // TODO: Check?
        edge_ids[pos] = edge_id as EdgeId;
        dests[pos] = dest;
        cursor[src as usize] += 1;
    }

    Ok(Csr {
        offsets,
        edge_ids,
        dests,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_empty_graph() {
        let csr = build_csr(3, &[]).unwrap();

        assert_eq!(
            csr,
            Csr {
                offsets: vec![0, 0, 0, 0],
                edge_ids: vec![],
                dests: vec![],
            }
        );
    }

    #[test]
    fn builds_csr_grouped_by_src() {
        let csr = build_csr(4, &[(0, 1), (2, 3), (0, 2), (3, 0)]).unwrap();

        assert_eq!(csr.offsets, vec![0, 2, 2, 3, 4]);
        assert_eq!(csr.edge_ids, vec![0, 2, 1, 3]);
        assert_eq!(csr.dests, vec![1, 2, 3, 0]);
    }
}
